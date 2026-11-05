// Loads the real data sets of the explainers (xp/data/NAME.json, written by xp/prep/*.py).
// Arrays are stored as {"shape": [...], "dtype": "f4" | "u2" | "u1", "b64": "..."}: little-endian, base64.
const cache = new Map();
const CTOR = { f4: Float32Array, u2: Uint16Array, u1: Uint8Array, i2: Int16Array };

function decode(v) {
  if (v && typeof v === "object" && "b64" in v && "dtype" in v) {
    const bin = atob(v.b64), buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    const a = new CTOR[v.dtype](buf.buffer);
    a.shape = v.shape;
    return a;
  }
  if (Array.isArray(v)) return v.map(decode);
  if (v && typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, decode(x)]));
  return v;
}

export function loadData(name) {
  if (!cache.has(name)) {
    const url = new URL(`./data/${name}.json`, import.meta.url);
    cache.set(name, fetch(url).then((r) => { if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`); return r.json(); }).then(decode));
  }
  return cache.get(name);
}
