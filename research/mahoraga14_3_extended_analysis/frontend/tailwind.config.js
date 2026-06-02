/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Arial", "sans-serif"],
      },
      colors: {
        ink: "#15181d",
        muted: "#68707d",
        line: "#dfe3e8",
        panel: "#f7f8fa",
        accent: "#1d4ed8",
        risk: "#b42318",
      },
    },
  },
  plugins: [],
};
