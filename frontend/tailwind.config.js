/** @type {import('tailwindcss').Config} */

// Theme colors are CSS variables (see src/index.css) so dark mode is a single
// `dark` class on <html>.
const token = (name) => `var(--${name})`;

export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: Object.fromEntries(
        [
          'page', 'surface', 'subtle', 'sunken', 'line', 'line-strong', 'ink', 'muted', 'muted-2',
          'faint', 'link', 'btn', 'btn-fg', 'glass', 'glass-soft', 'glass-line', 'hero-from',
          'hero-via', 'hero-to', 'hero-hl', 'watermark', 'warn-bg', 'warn-line', 'warn-ink', 'danger',
          'danger-bg', 'danger-line',
          'how-bg', 'how-card', 'how-art-from', 'how-art-to', 'stage',
        ].map((name) => [name, token(name === 'page' ? 'bg' : name)])
      ),
    },
  },
  plugins: [],
}
