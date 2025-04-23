module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx}',
    './src/components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3b82f6',
        secondary: '#10b981',
        accent: '#8b5cf6',
        danger: '#ef4444',
        // Rename text colors to avoid confusion with utility class names
        dark: '#1f2937',      // Very dark gray, almost black
        medium: '#4b5563',    // Medium gray for secondary text
        light: '#9ca3af',     // Light gray for tertiary text
      },
    },
  },
  plugins: [],
}