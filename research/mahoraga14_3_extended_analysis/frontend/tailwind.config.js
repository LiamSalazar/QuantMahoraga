/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Arial", "sans-serif"],
      },
      colors: {
        base: "#070809",
        ink: "#e7eaee",
        muted: "#8c949e",
        line: "#242a31",
        panel: "#0f1215",
        "panel-strong": "#15191e",
        accent: "#91a9b8",
        risk: "#b77b72",
      },
    },
  },
  plugins: [],
};
