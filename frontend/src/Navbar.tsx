import React, { useSyncExternalStore } from 'react';

export type View = 'landing' | 'analyzer';
export type Navigate = (view: View, anchor?: string) => void;

// Shared sticky top bar for the landing page and the analyzer: the logo goes
// home, then the theme toggle and the page-specific `action` button sit on
// the right.
export default function Navbar({ onNavigate, action }: { onNavigate: Navigate; action?: React.ReactNode }) {
  return (
    <header className="sticky top-0 z-40 bg-glass backdrop-blur-md border-b border-line text-ink">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        <button onClick={() => onNavigate('landing')} className="flex items-center gap-2.5 shrink-0" aria-label="ESMA Dental home">
          <span className="w-9 h-9 rounded-xl bg-btn text-btn-fg flex items-center justify-center">
            <ToothIcon />
          </span>
          <span className="text-[17px] font-semibold tracking-tight font-['IBM_Plex_Sans',sans-serif]">ESMA Dental</span>
        </button>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          {action}
        </div>
      </div>
    </header>
  );
}

// The theme lives on <html class="dark"> (index.html sets it before first
// paint), so every navbar instance reads it from there and stays in sync.
const subscribeTheme = (onChange: () => void) => {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  return () => observer.disconnect();
};
const isDarkTheme = () => document.documentElement.classList.contains('dark');

function ThemeToggle() {
  const dark = useSyncExternalStore(subscribeTheme, isDarkTheme);
  const toggle = () => {
    document.documentElement.classList.toggle('dark', !dark);
    try {
      localStorage.setItem('theme', dark ? 'light' : 'dark');
    } catch {
      // storage blocked: the choice just won't persist
    }
  };
  return (
    <button
      onClick={toggle}
      aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      title={dark ? 'Light mode' : 'Dark mode'}
      className="neu-sm w-[38px] h-[38px] rounded-lg inline-flex items-center justify-center
                 text-muted hover:text-ink transition-colors"
    >
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9"
        strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        {dark ? (
          <path d="M12 3v2M12 19v2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M3 12h2M19 12h2M5.6 18.4L7 17M17 7l1.4-1.4M12 8a4 4 0 100 8 4 4 0 000-8z" />
        ) : (
          <path d="M20 14.5A8 8 0 019.5 4a8 8 0 1010.5 10.5z" />
        )}
      </svg>
    </button>
  );
}

export const NAV_BUTTON =
  'neu-btn min-h-[38px] px-4 rounded-lg bg-btn text-btn-fg text-[13px] font-semibold hover:bg-[var(--btn-hover)] transition-colors whitespace-nowrap';

export function ToothIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 3c-2.5 0-4 2-4 4.5 0 3 1.5 4.5 2 7.5.4 2.6 1 6 2.6 6 1.4 0 1.6-3 2.4-5 .4-1 1.6-1 2 0 .8 2 1 5 2.4 5 1.6 0 2.2-3.4 2.6-6 .5-3 2-4.5 2-7.5C21 5 19.5 3 17 3c-2 0-3 1-5 1S9 3 7 3z" />
    </svg>
  );
}
