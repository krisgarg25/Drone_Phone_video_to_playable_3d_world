export const dynamic = "force-dynamic";

async function forward(request: Request) {
  const url = new URL(request.url);
  const path = url.pathname.replace(/^\/api\/backend/, "");
  if (!/^\/api\/(workspace\/(?:projects|project|measurements(?:\/delete)?|placements(?:\/(?:delete|update))?|file|run|cancel|upload)|survey(?:\/[^/]*)?|info|status|presets|scenes|tail|run|kill|upload)$/.test(path)) {
    return Response.json({ error: "Unknown API route" }, { status: 404 });
  }
  const writing = !["GET", "HEAD"].includes(request.method);
  if (writing) {
    let trusted = false;
    try {
      const authority = new URL(`${url.protocol}//${request.headers.get("host")}`);
      const origin = new URL(request.headers.get("origin") ?? "");
      trusted = ["localhost", "127.0.0.1", "[::1]"].includes(authority.hostname)
        && origin.origin === authority.origin && origin.pathname === "/" && !origin.search && !origin.hash;
    } catch { trusted = false; }
    if (!trusted) return Response.json({ error: "Requests must originate from this local application." }, { status: 403 });
  }
  const backend = process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8137";
  const headers = new Headers({ "accept-encoding": "identity" });
  for (const key of ["content-type", "content-length", "range"]) {
    const value = request.headers.get(key);
    if (value) headers.set(key, value);
  }
  if (writing) headers.set("origin", new URL(backend).origin);
  try {
    const upstream = await fetch(`${backend}${path}${url.search}`, {
      method: request.method, headers, body: writing ? request.body : undefined,
      duplex: "half", cache: "no-store", signal: request.signal,
    } as RequestInit & { duplex: "half" });
    const responseHeaders = new Headers({ "cache-control": "no-store", "x-content-type-options": "nosniff" });
    for (const key of ["content-type", "content-length", "content-range", "accept-ranges", "content-disposition"]) {
      const value = upstream.headers.get(key);
      if (value) responseHeaders.set(key, value);
    }
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch {
    return Response.json({ error: "Reconstruction service is offline. Start the local Python service and retry." }, { status: 502 });
  }
}

export const GET = forward;
export const HEAD = forward;
export const POST = forward;
