export function SelectControl({
  label,
  value,
  options,
  onChange,
  compact = false,
}: {
  label: string;
  value: string | number;
  options: (string | number)[];
  onChange: (value: string) => void;
  compact?: boolean;
}) {
  return (
    <label className={`control ${compact ? "compact" : ""}`}>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={String(option)} value={String(option)}>
            {String(option)}
          </option>
        ))}
      </select>
    </label>
  );
}

export function SliderControl({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void }) {
  return (
    <label className="slider-control">
      <span>
        {label}
        <b>{value.toFixed(2)}</b>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}
