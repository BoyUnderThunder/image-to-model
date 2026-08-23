"""Write a self-contained HTML viewer for a mesh.

The page embeds the geometry and texture directly and draws them with raw
WebGL, so it is a single file with no external requests and no libraries — it
works offline, over a file:// URL, or attached to a message. Handy for checking
a model before spending time importing it into Studio.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .types import Mesh

__all__ = ["write_viewer", "viewer_html"]


def _b64_array(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def _b64_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB" if array.shape[2] == 3 else "RGBA").save(
        buffer, format="PNG", optimize=True
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; height:100vh; overflow:hidden; background:#15171c;
         font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color:#e6e8ee; }
  canvas { display:block; width:100%; height:100%; cursor:grab; }
  canvas:active { cursor:grabbing; }
  .panel { position:fixed; top:16px; left:16px; padding:12px 16px; border-radius:10px;
           background:rgba(20,22,28,.82); border:1px solid rgba(255,255,255,.12);
           font-size:13px; line-height:1.65; backdrop-filter:blur(8px); }
  .panel h1 { margin:0 0 6px; font-size:14px; font-weight:600; }
  .panel dt { color:#9aa2b4; display:inline; }
  .panel dd { display:inline; margin:0 0 0 6px; font-variant-numeric:tabular-nums; }
  .panel div { white-space:nowrap; }
  .hint { position:fixed; bottom:16px; left:16px; font-size:12px; color:#8b93a5; }
  .error { position:fixed; inset:0; display:grid; place-content:center; text-align:center; padding:32px; }
</style>
</head>
<body>
<canvas id="c"></canvas>
<div class="panel">
  <h1>__TITLE__</h1>
  <div><dt>Triangles</dt><dd>__FACES__</dd></div>
  <div><dt>Vertices</dt><dd>__VERTS__</dd></div>
  <div><dt>Size</dt><dd>__SIZE__</dd></div>
</div>
<div class="hint">Drag to orbit &middot; scroll to zoom &middot; double-click to reset</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById('payload').textContent);
  const decode = (b64, Type) => {
    const bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Type(bytes.buffer);
  };

  const canvas = document.getElementById('c');
  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  if (!gl) {
    document.body.innerHTML = '<div class="error">This browser has no WebGL support.</div>';
    return;
  }
  const isGL2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
  if (!isGL2) gl.getExtension('OES_element_index_uint');

  const positions = decode(data.positions, Float32Array);
  const normals = decode(data.normals, Float32Array);
  const indices = decode(data.indices, Uint32Array);
  const uvs = data.uvs ? decode(data.uvs, Float32Array) : null;
  const colors = data.colors ? decode(data.colors, Uint8Array) : null;

  const vertexSrc = `
    attribute vec3 aPos; attribute vec3 aNormal; attribute vec2 aUv; attribute vec3 aColor;
    uniform mat4 uProj, uView; uniform mat3 uNormalMat;
    varying vec3 vNormal; varying vec2 vUv; varying vec3 vColor;
    void main() {
      vNormal = normalize(uNormalMat * aNormal);
      vUv = aUv; vColor = aColor;
      gl_Position = uProj * uView * vec4(aPos, 1.0);
    }`;

  const fragmentSrc = `
    precision mediump float;
    uniform sampler2D uTex; uniform bool uUseTex;
    varying vec3 vNormal; varying vec2 vUv; varying vec3 vColor;
    void main() {
      vec3 base = uUseTex ? texture2D(uTex, vUv).rgb : vColor;
      vec3 n = normalize(vNormal);
      // Two lights plus ambient, so faces turned away still read.
      float key = max(dot(n, normalize(vec3(-0.4, 0.6, 0.8))), 0.0);
      float fill = max(dot(n, normalize(vec3(0.5, -0.2, -0.6))), 0.0) * 0.28;
      gl_FragColor = vec4(base * (0.24 + 0.76 * key + fill), 1.0);
    }`;

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSrc));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSrc));
  gl.linkProgram(program);
  gl.useProgram(program);

  function bindAttrib(name, array, size, type, normalize) {
    const loc = gl.getAttribLocation(program, name);
    if (loc < 0) return;
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, type, normalize, 0, 0);
  }
  const vertexCount = positions.length / 3;
  bindAttrib('aPos', positions, 3, gl.FLOAT, false);
  bindAttrib('aNormal', normals, 3, gl.FLOAT, false);
  bindAttrib('aUv', uvs || new Float32Array(vertexCount * 2), 2, gl.FLOAT, false);
  bindAttrib('aColor', colors || new Uint8Array(vertexCount * 3).fill(200), 3, gl.UNSIGNED_BYTE, true);

  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

  let useTexture = false;
  if (data.texture) {
    const texture = gl.createTexture();
    const image = new Image();
    image.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      useTexture = true; draw();
    };
    image.src = 'data:image/png;base64,' + data.texture;
  }

  const uProj = gl.getUniformLocation(program, 'uProj');
  const uView = gl.getUniformLocation(program, 'uView');
  const uNormalMat = gl.getUniformLocation(program, 'uNormalMat');
  const uUseTex = gl.getUniformLocation(program, 'uUseTex');

  const center = data.center, radius = Math.max(data.radius, 1e-4);
  const home = { yaw: 0.6, pitch: 0.3, dist: radius * 3.2 };
  let cam = Object.assign({}, home);

  function draw() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.082, 0.090, 0.110, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const aspect = w / Math.max(h, 1), f = 1 / Math.tan(0.5 * 0.9);
    const near = radius * 0.02, far = radius * 40;
    const proj = [f / aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0];

    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    const eye = [center[0] + cam.dist*cp*sy, center[1] + cam.dist*sp, center[2] + cam.dist*cp*cy];

    let zx = eye[0]-center[0], zy = eye[1]-center[1], zz = eye[2]-center[2];
    let zl = Math.hypot(zx,zy,zz) || 1; zx/=zl; zy/=zl; zz/=zl;
    let xx = zz, xy = 0, xz = -zx;
    const xl = Math.hypot(xx,xy,xz) || 1; xx/=xl; xy/=xl; xz/=xl;
    const yx = zy*xz - zz*xy, yy = zz*xx - zx*xz, yz = zx*xy - zy*xx;
    const view = [xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0,
      -(xx*eye[0]+xy*eye[1]+xz*eye[2]), -(yx*eye[0]+yy*eye[1]+yz*eye[2]), -(zx*eye[0]+zy*eye[1]+zz*eye[2]), 1];

    gl.uniformMatrix4fv(uProj, false, new Float32Array(proj));
    gl.uniformMatrix4fv(uView, false, new Float32Array(view));
    gl.uniformMatrix3fv(uNormalMat, false, new Float32Array([1,0,0, 0,1,0, 0,0,1]));
    gl.uniform1i(uUseTex, useTexture ? 1 : 0);
    gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0);
  }

  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointerup', e => { dragging = false; canvas.releasePointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (!dragging) return;
    cam.yaw -= (e.clientX - lastX) * 0.01;
    cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + (e.clientY - lastY) * 0.01));
    lastX = e.clientX; lastY = e.clientY; draw();
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    cam.dist = Math.max(radius * 0.4, Math.min(radius * 20, cam.dist * Math.exp(e.deltaY * 0.001)));
    draw();
  }, { passive: false });
  canvas.addEventListener('dblclick', () => { cam = Object.assign({}, home); draw(); });
  window.addEventListener('resize', draw);
  draw();
})();
</script>
</body>
</html>
"""


def viewer_html(mesh: Mesh, texture: np.ndarray | None = None, title: str = "3D model") -> str:
    """Build the complete HTML page for ``mesh`` as a string."""
    prepared = mesh if mesh.vertex_normals is not None else mesh.with_normals()

    low, high = prepared.bounds()
    center = ((low + high) * 0.5).tolist()
    radius = float(np.linalg.norm(high - low)) * 0.5

    payload = {
        "positions": _b64_array(prepared.vertices, "<f4"),
        "normals": _b64_array(prepared.vertex_normals, "<f4"),
        "indices": _b64_array(prepared.faces, "<u4"),
        "uvs": _b64_array(prepared.uvs, "<f4") if prepared.uvs is not None else None,
        "colors": _b64_array(prepared.vertex_colors, "u1")
        if prepared.vertex_colors is not None
        else None,
        "texture": _b64_png(texture) if texture is not None else None,
        "center": center,
        "radius": radius,
    }

    extent = prepared.extent()
    replacements = {
        "__TITLE__": title,
        "__FACES__": f"{prepared.n_faces:,}",
        "__VERTS__": f"{prepared.n_vertices:,}",
        "__SIZE__": f"{extent[0]:.2f} &times; {extent[1]:.2f} &times; {extent[2]:.2f}",
        "__PAYLOAD__": json.dumps(payload),
    }
    page = _PAGE
    for placeholder, value in replacements.items():
        page = page.replace(placeholder, value)
    return page


def write_viewer(
    mesh: Mesh,
    path: str | os.PathLike,
    texture: np.ndarray | None = None,
    title: str | None = None,
) -> Path:
    """Write the viewer page to ``path`` and return it."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        viewer_html(mesh, texture=texture, title=title or out_path.stem), encoding="utf-8"
    )
    return out_path
