import type { Prediction } from "../api";

const pln = (n: number) => new Intl.NumberFormat("pl-PL", { maximumFractionDigits: 0 }).format(n) + " zł";

interface Props {
  title: string;
  suffix?: string;
  p: Prediction;
  r2?: number;
}

export default function ResultCard({ title, suffix = "", p, r2 }: Props) {
  return (
    <div className="result">
      <div className="result-title">{title}</div>
      <div className="result-price">{pln(p.price)}<span>{suffix}</span></div>
      <div className="result-sub">{pln(p.pricePerM2)} / m²</div>
      <div className="result-range">
        Typical range {pln(p.low)} – {pln(p.high)}
        <span className="muted"> (± avg. error{r2 !== undefined ? `, R² ${r2.toFixed(2)}` : ""})</span>
      </div>
    </div>
  );
}
