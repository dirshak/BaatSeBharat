import { useState } from 'react'

export default function Tabs({ tabs, initial = 0 }) {
  // tabs: [{ label, content }]
  const [active, setActive] = useState(initial)
  return (
    <div>
      <div className="tabs">
        {tabs.map((t, i) => (
          <button key={i} className={i === active ? 'active' : ''} onClick={() => setActive(i)}>
            {t.label}
          </button>
        ))}
      </div>
      <div>{tabs[active].content}</div>
    </div>
  )
}
