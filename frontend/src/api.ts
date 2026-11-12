export type Target = "rent" | "sale";

export interface CityMeta {
  centre: [number, number];
  medians: Record<string, number>;
  count: number;
}

export interface Meta {
  categories: Record<"city" | "type" | "ownership" | "buildingMaterial" | "condition", string[]>;
  cities: Record<string, CityMeta>;
  targets: Record<Target, { mae: number; r2: number; trainedAt?: string }>;
}

export interface Prediction {
  price: number;
  pricePerM2: number;
  low: number;
  high: number;
  mae: number;
}

export interface PredictResponse {
  rent?: Prediction;
  sale?: Prediction;
  imputed: Record<string, number>;
  predictor: string;
}

export type FlatInput = Record<string, string | number | boolean | null | string[]>;

const BASE = import.meta.env.VITE_API_URL ?? "";

export async function fetchMeta(): Promise<Meta> {
  const r = await fetch(`${BASE}/api/meta`);
  if (!r.ok) throw new Error(`Failed to load metadata (${r.status})`);
  return r.json();
}

export async function predict(body: FlatInput): Promise<PredictResponse> {
  const r = await fetch(`${BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d: { loc?: string[]; msg: string }) => `${d.loc?.slice(-1)[0] ?? ""}: ${d.msg}`).join("; ")
      : data.detail ?? `Request failed (${r.status})`;
    throw new Error(detail);
  }
  return data;
}
