"""MISB ST 0601 KLV telemetry from STANAG 4609 video (MIL-16, BOR-07).

Military and many border ISR cameras write their own position and attitude into the video
as KLV metadata inside the MPEG-2 transport stream. This module reads it without FFmpeg:

1. **Transport stream.** 188-byte packets (sync 0x47). The PAT names the PMT; the PMT
   names the metadata PID (stream_type 0x15, or 0x06 private data carrying a 'KLVA'
   registration). Without a usable PMT every PID's PES payloads are scanned for the key.
2. **PES.** Payloads are reassembled per PID from ``payload_unit_start`` to the next;
   the 33-bit PTS (90 kHz) timestamps each packet - the *video* clock.
3. **UAS Datalink Local Set.** 16-byte universal key 06 0E 2B 34 02 0B 01 01 0E 01 03
   01 01 00 00 00, BER length, then tag-length-value items (BER-OID tags). Tag 1 is a
   CRC-16-CCITT over everything from the key to the checksum's own length byte; a
   packet whose checksum does not match is rejected, never half-used.

Decoded (ST 0601.17 mappings): 2 precision time stamp (us since 1970), 5/6/7 platform
heading / pitch / roll, 13/14 sensor latitude / longitude, 15 sensor true altitude (MSL),
75 sensor ellipsoid height (HAE - preferred when present, it is the datum survey_georef
accepts), 16/17 sensor horizontal / vertical field of view, 18/19/20 sensor relative
azimuth / elevation / roll, 23/24 frame centre latitude / longitude. Out-of-range markers
(the ST 0601 "error" values) are dropped per field.

``parse_klv`` returns rows in the survey_inputs contract (``t_sec`` on the video clock,
lat, lon, altitude) and records gimbal attitude (sensor elevation as pitch, relative roll
+ platform roll as roll) for the straight-track alignment (M11). KLV carries no fix
quality, so uncertainty must be declared, as for a DJI subtitle.
"""
import struct
from pathlib import Path

UAS_LS_KEY = bytes.fromhex("060E2B34020B01010E01030101000000")
TS_PACKET = 188


# ------------------------------------------------------------------ BER / CRC helpers
def ber_length(data, i):
    first = data[i]
    if first < 0x80:
        return first, i + 1
    n = first & 0x7F
    if n == 0 or n > 4 or i + 1 + n > len(data):
        raise ValueError("invalid BER length")
    return int.from_bytes(data[i + 1:i + 1 + n], "big"), i + 1 + n


def ber_oid(data, i):
    value = 0
    while True:
        if i >= len(data):
            raise ValueError("truncated BER-OID tag")
        byte = data[i]
        value = (value << 7) | (byte & 0x7F)
        i += 1
        if not byte & 0x80:
            return value, i


def crc16_ccitt(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def _ber(n):
    if n < 128:
        return bytes([n])
    body = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(body)]) + body


# ------------------------------------------------------------------ ST 0601 field maps
def _scale(raw, bits, signed, lo, hi):
    if signed:
        full = 2 ** (bits - 1) - 1
        if raw == -(2 ** (bits - 1)):            # ST 0601 reserved "out of range"
            return None
        return raw * (hi - lo) / 2 / full
    return lo + raw * (hi - lo) / (2 ** bits - 1)


FIELDS = {
    # tag: (name, struct code, bits, signed, lo, hi)
    5: ("platform_heading_deg", ">H", 16, False, 0.0, 360.0),
    6: ("platform_pitch_deg", ">h", 16, True, -20.0, 20.0),
    7: ("platform_roll_deg", ">h", 16, True, -50.0, 50.0),
    13: ("sensor_latitude_deg", ">i", 32, True, -90.0, 90.0),
    14: ("sensor_longitude_deg", ">i", 32, True, -180.0, 180.0),
    15: ("sensor_msl_m", ">H", 16, False, -900.0, 19000.0),
    16: ("sensor_hfov_deg", ">H", 16, False, 0.0, 180.0),
    17: ("sensor_vfov_deg", ">H", 16, False, 0.0, 180.0),
    18: ("sensor_rel_azimuth_deg", ">I", 32, False, 0.0, 360.0),
    19: ("sensor_rel_elevation_deg", ">i", 32, True, -180.0, 180.0),
    20: ("sensor_rel_roll_deg", ">I", 32, False, 0.0, 360.0),
    23: ("frame_centre_latitude_deg", ">i", 32, True, -90.0, 90.0),
    24: ("frame_centre_longitude_deg", ">i", 32, True, -180.0, 180.0),
    75: ("sensor_hae_m", ">H", 16, False, -900.0, 19000.0),
}


def decode_local_set(packet):
    """One UAS LS (key included) -> dict of decoded fields. Raises on a bad checksum."""
    if not packet.startswith(UAS_LS_KEY):
        raise ValueError("not a UAS Datalink Local Set")
    length, start = ber_length(packet, len(UAS_LS_KEY))
    end = start + length
    if end > len(packet):
        raise ValueError("truncated local set")
    out, i, checksum = {}, start, None
    while i < end:
        tag, i = ber_oid(packet, i)
        n, i = ber_length(packet, i)
        value = packet[i:i + n]
        if tag == 1:
            if n != 2:
                raise ValueError("checksum item must be two bytes")
            checksum = (int.from_bytes(value, "big"), i)
        elif tag == 2 and n == 8:
            out["time_us"] = int.from_bytes(value, "big")
        elif tag in FIELDS:
            name, code, bits, signed, lo, hi = FIELDS[tag]
            if n == struct.calcsize(code):
                scaled = _scale(struct.unpack(code, value)[0], bits, signed, lo, hi)
                if scaled is not None:
                    out[name] = scaled
        i += n
    if checksum is None:
        raise ValueError("local set has no checksum (tag 1)")
    stored, value_at = checksum
    if crc16_ccitt(packet[:value_at]) != stored:
        raise ValueError("local set checksum mismatch")
    return out


def encode_local_set(fields):
    """Inverse of ``decode_local_set`` for tests and synthetic streams."""
    body = b""
    if "time_us" in fields:
        body += bytes([2, 8]) + int(fields["time_us"]).to_bytes(8, "big")
    for tag, (name, code, bits, signed, lo, hi) in FIELDS.items():
        if name not in fields:
            continue
        v = fields[name]
        if signed:
            raw = int(round(v / ((hi - lo) / 2) * (2 ** (bits - 1) - 1)))
        else:
            raw = int(round((v - lo) / (hi - lo) * (2 ** bits - 1)))
        body += bytes([tag, struct.calcsize(code)]) + struct.pack(code, raw)
    body += bytes([1, 2])
    head = UAS_LS_KEY + _ber(len(body) + 2)
    crc = crc16_ccitt(head + body)
    return head + body + crc.to_bytes(2, "big")


# ------------------------------------------------------------------ transport stream
def _pes_payload(pes):
    """(pts_seconds or None, payload bytes) of one PES packet."""
    if len(pes) < 9 or pes[:3] != b"\x00\x00\x01":
        return None, b""
    stream_id = pes[3]
    if stream_id in (0xBC, 0xBE, 0xBF, 0xF0, 0xF1, 0xFF, 0xF2, 0xF8):
        return None, pes[6:]
    flags, header_len = pes[7], pes[8]
    pts = None
    if flags & 0x80 and len(pes) >= 14:
        b = pes[9:14]
        pts = (((b[0] >> 1) & 0x07) << 30 | b[1] << 22 | (b[2] >> 1) << 15 | b[3] << 7 | b[4] >> 1) / 90000.0
    return pts, pes[9 + header_len:]


def _sections(payload, start):
    if start:
        pointer = payload[0]
        return payload[1 + pointer:]
    return payload


def _metadata_pids(packets):
    pat_pmts, meta = set(), set()
    for pid, start, payload in packets:
        if pid == 0 and start:
            sec = _sections(payload, True)
            length = ((sec[1] & 0x0F) << 8) | sec[2]
            for k in range(8, 3 + length - 4, 4):
                program = (sec[k] << 8) | sec[k + 1]
                if program:
                    pat_pmts.add(((sec[k + 2] & 0x1F) << 8) | sec[k + 3])
    for pid, start, payload in packets:
        if pid in pat_pmts and start:
            sec = _sections(payload, True)
            length = ((sec[1] & 0x0F) << 8) | sec[2]
            info_len = ((sec[10] & 0x0F) << 8) | sec[11]
            k = 12 + info_len
            while k < 3 + length - 4:
                stype = sec[k]
                epid = ((sec[k + 1] & 0x1F) << 8) | sec[k + 2]
                es_len = ((sec[k + 3] & 0x0F) << 8) | sec[k + 4]
                desc = sec[k + 5:k + 5 + es_len]
                if stype == 0x15 or (stype == 0x06 and b"KLVA" in desc):
                    meta.add(epid)
                k += 5 + es_len
    return meta


def read_klv_packets(data):
    """[(pts_seconds, local-set bytes)] from a transport stream's metadata PES."""
    if len(data) < TS_PACKET or data[0] != 0x47:
        raise ValueError("not an MPEG transport stream (no 0x47 sync byte)")
    packets = []
    for off in range(0, len(data) - TS_PACKET + 1, TS_PACKET):
        pkt = data[off:off + TS_PACKET]
        if pkt[0] != 0x47:
            raise ValueError(f"lost transport-stream sync at byte {off}")
        start = bool(pkt[1] & 0x40)
        pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
        afc = (pkt[3] >> 4) & 0x03
        i = 4
        if afc in (2, 3):
            i += 1 + pkt[4]
        if afc in (1, 3) and i < TS_PACKET:
            packets.append((pid, start, pkt[i:]))
    wanted = _metadata_pids(packets)
    streams, out = {}, []

    def flush(pid):
        buf = streams.pop(pid, None)
        if buf:
            pts, payload = _pes_payload(bytes(buf))
            at = payload.find(UAS_LS_KEY)
            while at >= 0:
                try:
                    length, body = ber_length(payload, at + len(UAS_LS_KEY))
                except ValueError:
                    break
                out.append((pts, payload[at:body + length]))
                at = payload.find(UAS_LS_KEY, body + length)

    for pid, start, payload in packets:
        if pid in (0, 0x1FFF) or (wanted and pid not in wanted):
            continue
        if start:
            flush(pid)
            streams[pid] = bytearray(payload)
        elif pid in streams:
            streams[pid] += payload
    for pid in list(streams):
        flush(pid)
    return out, bool(wanted)


def write_ts(local_sets, *, pid=0x101, pmt_pid=0x100, fps=10.0):
    """A minimal transport stream carrying ``local_sets`` as metadata PES (tests, demos)."""
    def packet(p, start, payload, cc):
        """One 188-byte packet; a short payload is padded with adaptation-field stuffing."""
        body, rest = payload[:184], payload[184:]
        flags = 0x40 if start else 0x00
        if len(body) == 184:
            return bytes([0x47, flags | (p >> 8), p & 0xFF, 0x10 | (cc & 0x0F)]) + body, rest
        stuff = 184 - len(body)
        adapt = bytes([stuff - 1]) + ((b"\x00" + b"\xff" * (stuff - 2)) if stuff > 1 else b"")
        return bytes([0x47, flags | (p >> 8), p & 0xFF, 0x30 | (cc & 0x0F)]) + adapt + body, rest

    def section(table_id, body):
        sec = bytes([table_id]) + (0xB000 | (len(body) + 5 + 4)).to_bytes(2, "big") + b"\x00\x01\xC1\x00\x00" + body
        crc = _crc32_mpeg(sec)
        return b"\x00" + sec + crc.to_bytes(4, "big")

    pat = section(0x00, (1).to_bytes(2, "big") + (0xE000 | pmt_pid).to_bytes(2, "big"))
    es = bytes([0x15]) + (0xE000 | pid).to_bytes(2, "big") + (0xF000).to_bytes(2, "big")
    pmt = section(0x02, (0xE000 | pid).to_bytes(2, "big") + (0xF000).to_bytes(2, "big") + es)
    out = bytearray()
    for p, table in ((0, pat), (pmt_pid, pmt)):
        out += packet(p, True, table, 0)[0]
    cc = 0
    for n, ls in enumerate(local_sets):
        ticks = int(round(n / fps * 90000))
        pts = bytes([0x21 | ((ticks >> 29) & 0x0E), (ticks >> 22) & 0xFF, 0x01 | ((ticks >> 14) & 0xFE),
                     (ticks >> 7) & 0xFF, 0x01 | ((ticks << 1) & 0xFE)])
        pes = b"\x00\x00\x01\xFC" + (len(ls) + 8).to_bytes(2, "big") + b"\x84\x80\x05" + pts + ls
        first = True
        while pes:
            chunk, pes = packet(pid, first, pes, cc)
            out += chunk
            first = False
            cc += 1
    return bytes(out)


def _crc32_mpeg(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else (crc << 1)
            crc &= 0xFFFFFFFF
    return crc


# ------------------------------------------------------------------ telemetry rows
def parse_klv(path, *, horizontal_std_m=None, vertical_std_m=None, strict=True):
    """Telemetry rows (survey_inputs contract) from a STANAG 4609 file's KLV."""
    import survey_inputs as ingest
    path = Path(path)
    packets, from_pmt = read_klv_packets(path.read_bytes())
    if not packets:
        raise ValueError(f"{path.name}: the transport stream carries no MISB ST 0601 local sets")
    rows, gimbal, rejected, warnings = [], [], [], []
    t0 = None
    for index, (pts, ls) in enumerate(packets):
        try:
            f = decode_local_set(ls)
        except ValueError as error:
            rejected.append({"block": index, "message": str(error)})
            continue
        if "sensor_latitude_deg" not in f or "sensor_longitude_deg" not in f:
            rejected.append({"block": index, "message": "no sensor latitude/longitude"})
            continue
        t = pts if pts is not None else (f["time_us"] / 1e6 if "time_us" in f else None)
        if t is None:
            rejected.append({"block": index, "message": "no PTS and no precision time stamp"})
            continue
        t0 = t if t0 is None else t0
        alt = f.get("sensor_hae_m", f.get("sensor_msl_m"))
        if alt is None:
            rejected.append({"block": index, "message": "no sensor altitude"})
            continue
        rows.append({"t_sec": round(t - t0, 6), "latitude_deg": f["sensor_latitude_deg"],
                     "longitude_deg": f["sensor_longitude_deg"], "altitude_m": alt})
        elevation = f.get("sensor_rel_elevation_deg")
        roll = f.get("platform_roll_deg")
        gimbal.append(None if elevation is None else {
            "pitch_deg": elevation + f.get("platform_pitch_deg", 0.0),
            "roll_deg": (roll or 0.0) + ((f.get("sensor_rel_roll_deg", 0.0) + 180.0) % 360.0 - 180.0),
            "yaw_deg": (f.get("platform_heading_deg", 0.0) + f.get("sensor_rel_azimuth_deg", 0.0)) % 360.0})
    if not rows:
        raise ValueError(f"{path.name}: no usable KLV packet ({rejected[0]['message'] if rejected else 'none'})")
    hae = all("sensor_hae_m" in decode_local_set(ls) for _, ls in packets[:1])
    if not hae:
        warnings.append("ST 0601 tag 15 is height above mean sea level; survey_georef needs ellipsoidal "
                        "heights, so a geoid separation must be applied (survey_geoid) first")
    if not from_pmt:
        warnings.append("no PMT named a metadata stream; KLV was found by scanning every PID")
    provenance = "none"
    if horizontal_std_m is not None:
        ingest._apply_declared_uncertainty(rows, horizontal_std_m, vertical_std_m)
        provenance = "declared_by_caller"
    elif strict:
        raise ValueError(f"{path.name}: " + ingest._UNCERTAINTY_HINT)
    ingest._record(path, dict(source="klv_st0601", time_reference="video",
                              altitude_datum="ellipsoidal" if hae else "mean_sea_level",
                              altitude_source="sensor_hae" if hae else "sensor_msl",
                              first_time_s=0.0, rows=len(rows), rejected=len(rejected),
                              gimbal=gimbal, uncertainty_source=provenance,
                              declared_std_m=ingest._declared_pair(provenance, horizontal_std_m, vertical_std_m),
                              warnings=warnings))
    return ingest._positional_only(rows) if provenance == "none" else ingest._six_fields(rows)
