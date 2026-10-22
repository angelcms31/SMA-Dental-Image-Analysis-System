import { useState } from 'react';
import LandingPage from './LandingPage.tsx';
import OpgAnalyzer from './OpgAnalyzer.tsx';
import type { View } from './Navbar.tsx';

function App() {
  const [view, setView] = useState<View>('landing');
  // Keep the analyzer mounted once opened so going back home doesn't lose results.
  const [started, setStarted] = useState(false);

  const navigate = (next: View, anchor?: string) => {
    if (next === 'analyzer') setStarted(true);
    setView(next);
    // wait for the new view to render before scrolling
    requestAnimationFrame(() => {
      const target = anchor ? document.getElementById(anchor) : null;
      if (target) target.scrollIntoView({ behavior: 'smooth' });
      else window.scrollTo(0, 0);
    });
  };

  return (
    <div>
      {view === 'landing' && <LandingPage onNavigate={navigate} />}
      {started && (
        <div hidden={view !== 'analyzer'}>
          <OpgAnalyzer onNavigate={navigate} />
        </div>
      )}
    </div>
  )
}

export default App;
