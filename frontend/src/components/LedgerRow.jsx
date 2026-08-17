export default function LedgerRow({ items }) {
  // items: [{ label, value, sub, tooltip }]
  return (
    <div className="ledger-row">
      {items.map((item, i) => (
        <div className="ledger-cell" key={i} title={item.tooltip || undefined}>
          <div className="ledger-label">{item.label}</div>
          <div className="ledger-value">{item.value}</div>
          {item.sub ? <div className="ledger-sub" dangerouslySetInnerHTML={{ __html: item.sub }} /> : null}
        </div>
      ))}
    </div>
  )
}
