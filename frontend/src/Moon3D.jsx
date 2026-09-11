import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

// Real lunar surface from NASA's CGI Moon Kit (SVS 4720):
// color = LROC WAC global mosaic, elevation = LOLA DEM. Bundled in
// public/textures so the hero works offline. Public domain (NASA).
const COLOR_URL = `${import.meta.env.BASE_URL}textures/moon-color-1k.jpg`
const ELEVATION_URL = `${import.meta.env.BASE_URL}textures/moon-elevation-1k.jpg`

// Real PSR-hosting craters at their true selenographic coordinates.
// PSRs cluster at the poles (axial tilt ~1.5°), matching the dark
// polar terrain in the LROC texture. Precision is ~1° — plenty for
// a 1K global map where 1px ≈ 0.35°.
// NOTE: the south-polar craters sit within ~5° of each other, which
// is a few pixels on a 260px disc, so each pill carries a small
// screen-space offset (ox/oy) to keep them readable. The blurbs give
// the true coordinates.
const MARKERS = [
  {
    id: 'shackleton',
    label: 'SHACKLETON',
    lat: -89.9,
    lon: 0.0,
    ox: -72,
    oy: -16,
    blurb: '89.9°S, 0°E · 21-km south-pole crater. Floor in permanent shadow, rim in near-eternal light. Prime Artemis water-ice target.',
  },
  {
    id: 'cabeus',
    label: 'CABEUS',
    lat: -85.3,
    lon: -35.5,
    ox: 68,
    oy: -12,
    blurb: '85.3°S, 35.5°W · LCROSS impacted here in 2009 and confirmed water ice in its shadowed floor.',
  },
  {
    id: 'shoemaker',
    label: 'SHOEMAKER',
    lat: -88.1,
    lon: 44.9,
    ox: -6,
    oy: 12,
    blurb: '88.1°S, 44.9°E · Shadowed south-polar floor. Lunar Prospector was directed into Shoemaker in 1999.',
  },
  {
    id: 'peary',
    label: 'PEARY',
    lat: 88.6,
    lon: 33.0,
    ox: 0,
    oy: 0,
    blurb: '88.6°N, 33°E · North-pole crater with permanently shadowed floor. Rotate the Moon to find it.',
  },
]

function latLonToVec3(lat, lon, radius) {
  const phi = ((90 - lat) * Math.PI) / 180
  const theta = ((lon + 180) * Math.PI) / 180
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  )
}

// Neutral-gray procedural fallback, used only if the NASA textures
// fail to load (e.g. offline preview of an old bundle).
function makeFallbackTextures() {
  const w = 512
  const h = 256
  const color = document.createElement('canvas')
  color.width = w
  color.height = h
  const bump = document.createElement('canvas')
  bump.width = w
  bump.height = h
  const c = color.getContext('2d')
  const b = bump.getContext('2d')
  c.fillStyle = '#83888f'
  c.fillRect(0, 0, w, h)
  b.fillStyle = '#808080'
  b.fillRect(0, 0, w, h)
  let seed = 987654321
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296
    return seed / 4294967296
  }
  for (let i = 0; i < 160; i++) {
    const x = rand() * w
    const y = rand() * h
    const r = 2 + Math.pow(rand(), 2) * 16
    c.fillStyle = 'rgba(70,72,78,0.45)'
    c.beginPath()
    c.arc(x, y, r, 0, Math.PI * 2)
    c.fill()
    c.strokeStyle = 'rgba(190,194,200,0.4)'
    c.lineWidth = 1
    c.beginPath()
    c.arc(x, y, r, Math.PI * 1.1, Math.PI * 1.9)
    c.stroke()
    b.fillStyle = 'rgba(60,60,60,0.6)'
    b.beginPath()
    b.arc(x, y, r * 0.8, 0, Math.PI * 2)
    b.fill()
  }
  const map = new THREE.CanvasTexture(color)
  map.colorSpace = THREE.SRGBColorSpace
  const bumpMap = new THREE.CanvasTexture(bump)
  return { map, bumpMap }
}

export default function Moon3D() {
  const mountRef = useRef(null)
  const markerRefs = useRef({})
  const [active, setActive] = useState(null)
  const [failed, setFailed] = useState(false)
  const activeRef = useRef(null)

  useEffect(() => {
    activeRef.current = active
  }, [active])

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    let renderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    } catch {
      setFailed(true)
      return
    }
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setClearColor(0x000000, 0)
    mount.appendChild(renderer.domElement)
    renderer.domElement.className = 'moon3d-canvas'

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100)
    camera.position.set(0, 0.15, 3.1)

    const group = new THREE.Group()
    scene.add(group)

    // Start on the procedural fallback so first paint never waits
    // on the network; swap to the NASA maps as soon as they arrive.
    const fallback = makeFallbackTextures()
    const moon = new THREE.Mesh(
      new THREE.SphereGeometry(1, 72, 72),
      new THREE.MeshStandardMaterial({
        map: fallback.map,
        bumpMap: fallback.bumpMap,
        bumpScale: 0.5,
        roughness: 1.0,
        metalness: 0.0,
      }),
    )
    group.add(moon)

    let cancelled = false
    const owned = [fallback.map, fallback.bumpMap]
    const loader = new THREE.TextureLoader()
    loader.setCrossOrigin('anonymous')
    const applyReal = (map, bumpMap) => {
      if (cancelled) {
        map?.dispose()
        bumpMap?.dispose()
        return
      }
      if (map) {
        map.colorSpace = THREE.SRGBColorSpace
        map.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy())
        moon.material.map = map
        owned.push(map)
      }
      if (bumpMap) {
        moon.material.bumpMap = bumpMap
        owned.push(bumpMap)
      }
      moon.material.needsUpdate = true
    }
    Promise.all([
      loader.loadAsync(COLOR_URL).catch(() => null),
      loader.loadAsync(ELEVATION_URL).catch(() => null),
    ]).then(([map, bumpMap]) => {
      if (map || bumpMap) applyReal(map, bumpMap)
    })

    // Neutral photographic lighting: warm-white key for a soft
    // terminator, faint neutral fill so the night side stays dark.
    const sun = new THREE.DirectionalLight(0xfff6ec, 2.4)
    sun.position.set(-3.0, 1.2, 2.4)
    scene.add(sun)
    scene.add(new THREE.AmbientLight(0x9aa0a8, 0.35))
    const fill = new THREE.DirectionalLight(0xdde4ee, 0.25)
    fill.position.set(2.5, -0.4, -2.0)
    scene.add(fill)

    // Starfield
    const starGeo = new THREE.BufferGeometry()
    const starCount = 350
    const pos = new Float32Array(starCount * 3)
    for (let i = 0; i < starCount; i++) {
      const r = 8 + Math.random() * 10
      const t = Math.random() * Math.PI * 2
      const p = Math.acos(2 * Math.random() - 1)
      pos[i * 3] = r * Math.sin(p) * Math.cos(t)
      pos[i * 3 + 1] = r * Math.sin(p) * Math.sin(t)
      pos[i * 3 + 2] = r * Math.cos(p)
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    scene.add(
      new THREE.Points(
        starGeo,
        new THREE.PointsMaterial({ color: 0xaac4dd, size: 0.035, transparent: true, opacity: 0.7 }),
      ),
    )

    // NOTE: three.js applies Euler XYZ as X-first, so rotation.x tips a
    // pole toward the viewer and rotation.y then spins longitudes past.
    // x=-0.9 tips the dark south-polar PSR terrain into view on load.
    group.rotation.set(-0.9, 0, 0.06)
    const target = { x: -0.9, y: 0 }
    const cur = { x: -0.9, y: 0 }
    let zoom = 3.1
    let zoomTarget = 3.1
    let dragging = false
    let lastX = 0
    let lastY = 0
    let velX = 0
    let velY = 0
    let lastInteract = 0
    let downX = 0
    let downY = 0

    const el = renderer.domElement
    const onDown = (e) => {
      dragging = true
      lastX = e.clientX
      lastY = e.clientY
      downX = e.clientX
      downY = e.clientY
      velX = 0
      velY = 0
      lastInteract = performance.now()
      el.setPointerCapture?.(e.pointerId)
    }
    const onMove = (e) => {
      if (!dragging) return
      const dx = e.clientX - lastX
      const dy = e.clientY - lastY
      lastX = e.clientX
      lastY = e.clientY
      target.y += dx * 0.005
      target.x += dy * 0.003
      target.x = Math.max(-1.1, Math.min(1.1, target.x))
      velY = dx * 0.005
      velX = dy * 0.003
      lastInteract = performance.now()
    }
    const onUp = (e) => {
      dragging = false
      lastInteract = performance.now()
      // treat as click on empty space -> clear active marker
      const moved = Math.hypot(e.clientX - downX, e.clientY - downY)
      if (moved < 6 && e.target === el) setActive(null)
    }
    const onWheel = (e) => {
      e.preventDefault()
      zoomTarget = Math.max(2.2, Math.min(4.4, zoomTarget + e.deltaY * 0.0016))
      lastInteract = performance.now()
    }
    el.addEventListener('pointerdown', onDown)
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    el.addEventListener('wheel', onWheel, { passive: false })

    const resize = () => {
      const w = mount.clientWidth || 260
      const h = mount.clientHeight || 260
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(mount)

    const v = new THREE.Vector3()
    const camDir = new THREE.Vector3()
    let raf = 0
    let last = performance.now()

    const tick = (now) => {
      raf = requestAnimationFrame(tick)
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now

      if (!dragging) {
        // inertia
        target.y += velY
        target.x = Math.max(-1.1, Math.min(1.1, target.x + velX))
        velY *= 0.94
        velX *= 0.94
        // idle auto-rotate (slow so the polar composition lingers)
        if (!reduceMotion && now - lastInteract > 2500) target.y += dt * 0.06
      }
      cur.x += (target.x - cur.x) * 0.08
      cur.y += (target.y - cur.y) * 0.08
      group.rotation.x = cur.x
      group.rotation.y = cur.y
      group.rotation.z = 0.06 + Math.sin(now * 0.0002) * 0.01

      zoom += (zoomTarget - zoom) * 0.1
      camera.position.z = zoom
      camera.lookAt(0, 0, 0)

      // Project markers
      const w = mount.clientWidth || 1
      const h = mount.clientHeight || 1
      camera.getWorldDirection(camDir)
      for (const m of MARKERS) {
        const node = markerRefs.current[m.id]
        if (!node) continue
        v.copy(latLonToVec3(m.lat, m.lon, 1.02)).applyEuler(group.rotation)
        const facing = v.clone().normalize().dot(camDir.clone().negate())
        const p = v.clone().project(camera)
        const x = (p.x * 0.5 + 0.5) * w
        const y = (-p.y * 0.5 + 0.5) * h
        const behind = facing < 0.18
        node.style.transform = `translate(${(x + (m.ox || 0)).toFixed(1)}px, ${(y + (m.oy || 0)).toFixed(1)}px) translate(-50%,-50%)`
        node.style.opacity = behind ? '0' : '1'
        node.style.pointerEvents = behind ? 'none' : 'auto'
        node.classList.toggle('is-active', activeRef.current === m.id)
      }

      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      cancelled = true
      cancelAnimationFrame(raf)
      ro.disconnect()
      el.removeEventListener('pointerdown', onDown)
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      el.removeEventListener('wheel', onWheel)
      moon.geometry.dispose()
      moon.material.dispose()
      for (const t of owned) t.dispose()
      starGeo.dispose()
      renderer.dispose()
      mount.removeChild(el)
    }
  }, [])

  if (failed) {
    return (
      <div className="moon3d-fallback" role="img" aria-label="Lunar map illustration">
        <div className="moon">
          <div className="terminator" />
        </div>
      </div>
    )
  }

  const activeMarker = MARKERS.find((m) => m.id === active)

  return (
    <div className="moon3d" role="application" aria-label="Interactive 3D moon. Drag to rotate, scroll to zoom.">
      <div ref={mountRef} className="moon3d-stage" />
      <div className="moon3d-markers" aria-hidden={false}>
        {MARKERS.map((m) => (
          <button
            key={m.id}
            ref={(n) => {
              if (n) markerRefs.current[m.id] = n
              else delete markerRefs.current[m.id]
            }}
            type="button"
            className="marker moon3d-marker"
            onClick={(e) => {
              e.stopPropagation()
              setActive((a) => (a === m.id ? null : m.id))
            }}
            title={m.blurb}
          >
            <i /> {m.label}
          </button>
        ))}
      </div>
      {activeMarker ? (
        <p className="moon3d-tip" role="status">
          <strong>{activeMarker.label}</strong> — {activeMarker.blurb}
        </p>
      ) : (
        <p className="moon3d-hint">Drag to rotate · Scroll to zoom · Click a marker</p>
      )}
      <p className="moon3d-credit">Moon texture: NASA LRO / LROC · LOLA</p>
    </div>
  )
}
