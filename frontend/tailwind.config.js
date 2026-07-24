/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#090d16',
        panel: 'rgba(15, 23, 42, 0.75)',
        border: 'rgba(255, 255, 255, 0.08)',
        cyan: {
          accent: '#00f2fe',
          glow: 'rgba(0, 242, 254, 0.25)',
        },
        amber: {
          accent: '#f59e0b',
        },
        emerald: {
          accent: '#10b981',
        },
        rose: {
          accent: '#f43f5e',
        }
      },
      fontFamily: {
        title: ['Outfit', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
