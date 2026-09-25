export const dynamic = "force-dynamic";

async function serve(request: Request, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  if (!path.length || !["viewer", "work", "videos"].includes(path[0]) || path.some((part) => !part || part === "." || part === ".." || /[\\/\x00]/.test(part))) {
    return Response.json({ error: "Asset not found" }, { status: 404 });
  }
  const url = new URL(request.url);
  const query = new URLSearchParams({ path: path.join("/") });
  if (url.searchParams.get("download") === "1") query.set("download", "1");
  const headers = new Headers({ "accept-encoding": "identity" });
  const range = request.headers.get("range");
  if (range) headers.set("range", range);
  try {
    const upstream = await fetch(`${process.env.BACKEND_ORIGIN ?? "http://127.0.0.1:8137"}/api/workspace/file?${query}`, {
      method: request.method, headers, cache: "no-store", signal: request.signal,
    });
    const output = new Headers({ "x-content-type-options": "nosniff", "cache-control": "no-store" });
    for (const name of ["content-type", "content-length", "content-range", "content-disposition", "accept-ranges"]) {
      const value = upstream.headers.get(name);
      if (value) output.set(name, value);
    }
    return new Response(upstream.body, { status: upstream.status, headers: output });
  } catch {
    return Response.json({ error: "The reconstruction service is offline." }, { status: 502 });
  }
}
export const GET = serve;
export const HEAD = serve;
