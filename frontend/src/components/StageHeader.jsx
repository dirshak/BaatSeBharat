export default function StageHeader({ number, title, subtitle }) {
  return (
    <>
      <div className="stage-eyebrow">Stage {number}</div>
      <h1>{title}</h1>
      {subtitle ? <p className="stage-subtitle">{subtitle}</p> : null}
    </>
  )
}
