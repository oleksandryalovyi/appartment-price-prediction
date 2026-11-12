import { useEffect, useMemo, useState } from "react";
import { fetchMeta, predict, type Meta, type PredictResponse } from "./api";
import { AMENITIES, CITY_LABELS, DISTANCE_FIELDS, label } from "./labels";
import MapPicker from "./components/MapPicker";
import ResultCard from "./components/ResultCard";

type Form = Record<string, string | boolean>;

const INITIAL: Form = {
  city: "warszawa", type: "blockOfFlats", squareMeters: "52", rooms: "2", floor: "3", floorCount: "5",
  buildYear: "1995", ownership: "condominium", buildingMaterial: "", condition: "",
  hasBalcony: true, hasElevator: false, hasParkingSpace: false, hasStorageRoom: false, hasSecurity: false,
  ...Object.fromEntries(DISTANCE_FIELDS.map((d) => [d.key, ""])),
};

function haversineKm(a: [number, number], b: [number, number]) {
  const r = (x: number) => (x * Math.PI) / 180;
  const h = Math.sin(r(b[0] - a[0]) / 2) ** 2 + Math.cos(r(a[0])) * Math.cos(r(b[0])) * Math.sin(r(b[1] - a[1]) / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

const num = (v: string | boolean) => (v === "" || typeof v === "boolean" ? null : Number(v));

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [form, setForm] = useState<Form>(INITIAL);
  const [point, setPoint] = useState<[number, number] | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { fetchMeta().then(setMeta).catch((e) => setError(e.message)); }, []);

  const city = meta?.cities[form.city as string];
  const centreDistance = point && city ? haversineKm(point, city.centre) : null;
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.type === "checkbox" ? (e.target as HTMLInputElement).checked : e.target.value }));

  const changeCity = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setForm((f) => ({ ...f, city: e.target.value }));
    setPoint(null);
    setResult(null);
  };

  const floorError = useMemo(() => {
    const f = num(form.floor), c = num(form.floorCount);
    return f !== null && c !== null && f > c ? "Floor can't be above the number of floors" : null;
  }, [form.floor, form.floorCount]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (floorError) return;
    setLoading(true); setError(null);
    try {
      const body: Record<string, unknown> = {
        city: form.city,
        type: form.type || null, ownership: form.ownership || null,
        buildingMaterial: form.buildingMaterial || null, condition: form.condition || null,
        squareMeters: num(form.squareMeters), rooms: num(form.rooms), floor: num(form.floor),
        floorCount: num(form.floorCount), buildYear: num(form.buildYear),
        latitude: point?.[0] ?? null, longitude: point?.[1] ?? null,
        centreDistance: centreDistance !== null ? +centreDistance.toFixed(2) : null,
        ...Object.fromEntries(DISTANCE_FIELDS.map((d) => [d.key, num(form[d.key])])),
        ...Object.fromEntries(AMENITIES.map((a) => [a.key, form[a.key]])),
      };
      setResult(await predict(body as never));
    } catch (err) {
      setError((err as Error).message); setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const select = (k: keyof Meta["categories"], allowEmpty = true) => (
    <select value={form[k] as string} onChange={k === "city" ? changeCity : set(k)} id={k}>
      {allowEmpty && <option value="">Unknown</option>}
      {meta?.categories[k].map((v) => <option key={v} value={v}>{label(v)}</option>)}
    </select>
  );

  const input = (k: string, props: React.InputHTMLAttributes<HTMLInputElement> = {}) => (
    <input id={k} type="number" inputMode="decimal" value={form[k] as string} onChange={set(k)} {...props} />
  );

  return (
    <div className="page">
      <header>
        <h1>Flat Price Analyzer</h1>
        <p className="muted">Estimate monthly rent and sale price for an apartment in 15 Polish cities. XGBoost models trained on ~130k Otodom-style listings (Aug 2023 – Jun 2024).</p>
      </header>

      {!meta && !error && <p className="muted">Loading model metadata…</p>}

      {meta && (
        <form className="layout" onSubmit={submit}>
          <div className="col">
            <section className="card">
              <h2>Basics</h2>
              <div className="grid">
                <label>City{select("city", false)}</label>
                <label>Building type{select("type")}</label>
                <label>Area, m²{input("squareMeters", { required: true, min: 10, max: 300, step: "0.1" })}</label>
                <label>Rooms{input("rooms", { required: true, min: 1, max: 10, step: 1 })}</label>
                <label>Floor{input("floor", { min: 0, max: 60, step: 1, placeholder: "0 = ground" })}</label>
                <label>Floors in building{input("floorCount", { min: 1, max: 60, step: 1 })}</label>
              </div>
              {floorError && <p className="error">{floorError}</p>}
            </section>

            <section className="card">
              <h2>Building</h2>
              <div className="grid">
                <label>Year built{input("buildYear", { min: 1800, max: 2030, step: 1 })}</label>
                <label>Ownership{select("ownership")}</label>
                <label>Material{select("buildingMaterial")}</label>
                <label>Condition{select("condition")}</label>
              </div>
              <div className="checks">
                {AMENITIES.map((a) => (
                  <label key={a.key} className="check">
                    <input type="checkbox" checked={form[a.key] as boolean} onChange={set(a.key)} />{a.label}
                  </label>
                ))}
              </div>
            </section>

            <details className="card">
              <summary><h2>Advanced: nearby places</h2><span className="muted">optional — city medians used when empty</span></summary>
              <div className="grid">
                {DISTANCE_FIELDS.map((d) => (
                  <label key={d.key}>{d.label}{d.unit && `, ${d.unit}`}
                    {input(d.key, { min: 0, step: "0.001", placeholder: city ? String(city.medians[d.key]) : "" })}
                  </label>
                ))}
              </div>
            </details>
          </div>

          <div className="col">
            <section className="card">
              <h2>Location <span className="muted small">click the map to drop a pin</span></h2>
              {city && <MapPicker centre={city.centre} value={point} onChange={setPoint} />}
              <p className="muted small">
                {point
                  ? <>Pin at {point[0].toFixed(4)}, {point[1].toFixed(4)} · {centreDistance!.toFixed(2)} km from {CITY_LABELS[form.city as string]} centre</>
                  : <>No pin — location is optional, but it improves the estimate.</>}
              </p>
            </section>

            <button type="submit" disabled={loading || !!floorError}>{loading ? "Estimating…" : "Estimate price"}</button>
            {error && <p className="error" role="alert">{error}</p>}

            {result && (
              <section className="results" aria-live="polite">
                {result.rent && <ResultCard title="Monthly rent" suffix=" / mo" p={result.rent} r2={meta.targets.rent?.r2} />}
                {result.sale && <ResultCard title="Sale price" p={result.sale} r2={meta.targets.sale?.r2} />}
                {Object.keys(result.imputed).length > 0 && (
                  <p className="muted small">Filled in from city data: {Object.keys(result.imputed).join(", ")}.</p>
                )}
              </section>
            )}
          </div>
        </form>
      )}
      {!meta && error && <p className="error">{error}</p>}
    </div>
  );
}
