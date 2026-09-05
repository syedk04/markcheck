// Runs inside the embed worker (OffscreenCanvas is available there in all
// browsers that support Web Workers well enough for transformers.js anyway).
// This is a deliberately lightweight stand-in for the offline pipeline's
// dedicated background-removal step (rembg): white-flatten alpha, then trim
// any near-uniform border. Good enough for typical logo uploads on a white
// or transparent background; busier photographic backgrounds will not be
// segmented as cleanly as the offline pipeline. Documented in README.

export interface PreparedImage {
  colorCanvas: OffscreenCanvas
  greyCanvas: OffscreenCanvas
  mask: Uint8Array
  maskWidth: number
  maskHeight: number
}

const BACKGROUND_THRESHOLD = 12
const MASK_SIZE = 160

export async function prepareImage(bitmap: ImageBitmap, targetSize: number): Promise<PreparedImage> {
  const flatCanvas = new OffscreenCanvas(bitmap.width, bitmap.height)
  const flatCtx = flatCanvas.getContext('2d')!
  flatCtx.fillStyle = '#ffffff'
  flatCtx.fillRect(0, 0, bitmap.width, bitmap.height)
  flatCtx.drawImage(bitmap, 0, 0)

  const { data } = flatCtx.getImageData(0, 0, bitmap.width, bitmap.height)
  const bbox = findContentBoundingBox(data, bitmap.width, bitmap.height)

  const cropW = bbox.maxX - bbox.minX + 1
  const cropH = bbox.maxY - bbox.minY + 1

  const squareSide = Math.max(cropW, cropH)
  const colorCanvas = new OffscreenCanvas(targetSize, targetSize)
  const colorCtx = colorCanvas.getContext('2d')!
  colorCtx.fillStyle = '#ffffff'
  colorCtx.fillRect(0, 0, targetSize, targetSize)
  const scale = targetSize / squareSide
  const destW = cropW * scale
  const destH = cropH * scale
  const destX = (targetSize - destW) / 2
  const destY = (targetSize - destH) / 2
  colorCtx.drawImage(flatCanvas, bbox.minX, bbox.minY, cropW, cropH, destX, destY, destW, destH)

  const greyCanvas = new OffscreenCanvas(targetSize, targetSize)
  const greyCtx = greyCanvas.getContext('2d')!
  greyCtx.drawImage(colorCanvas, 0, 0)
  const colorData = greyCtx.getImageData(0, 0, targetSize, targetSize)
  toGreyscaleInPlace(colorData.data)
  greyCtx.putImageData(colorData, 0, 0)

  const maskCanvas = new OffscreenCanvas(MASK_SIZE, MASK_SIZE)
  const maskCtx = maskCanvas.getContext('2d')!
  maskCtx.fillStyle = '#ffffff'
  maskCtx.fillRect(0, 0, MASK_SIZE, MASK_SIZE)
  maskCtx.drawImage(colorCanvas, 0, 0, targetSize, targetSize, 0, 0, MASK_SIZE, MASK_SIZE)
  const maskImageData = maskCtx.getImageData(0, 0, MASK_SIZE, MASK_SIZE)
  const mask = buildForegroundMask(maskImageData.data, MASK_SIZE, MASK_SIZE)

  return { colorCanvas, greyCanvas, mask, maskWidth: MASK_SIZE, maskHeight: MASK_SIZE }
}

function findContentBoundingBox(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = width
  let minY = height
  let maxX = -1
  let maxY = -1
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4
      const r = data[idx]
      const g = data[idx + 1]
      const b = data[idx + 2]
      const isBackground =
        Math.abs(r - 255) < BACKGROUND_THRESHOLD &&
        Math.abs(g - 255) < BACKGROUND_THRESHOLD &&
        Math.abs(b - 255) < BACKGROUND_THRESHOLD
      if (!isBackground) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
  }
  if (maxX === -1) {
    // Entirely blank upload; fall back to the full frame rather than crash.
    return { minX: 0, minY: 0, maxX: width - 1, maxY: height - 1 }
  }
  return { minX, minY, maxX, maxY }
}

function toGreyscaleInPlace(data: Uint8ClampedArray): void {
  for (let i = 0; i < data.length; i += 4) {
    const luma = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
    data[i] = luma
    data[i + 1] = luma
    data[i + 2] = luma
  }
}

function buildForegroundMask(data: Uint8ClampedArray, width: number, height: number): Uint8Array {
  const mask = new Uint8Array(width * height)
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4
    const r = data[idx]
    const g = data[idx + 1]
    const b = data[idx + 2]
    const isBackground =
      Math.abs(r - 255) < BACKGROUND_THRESHOLD &&
      Math.abs(g - 255) < BACKGROUND_THRESHOLD &&
      Math.abs(b - 255) < BACKGROUND_THRESHOLD
    mask[i] = isBackground ? 0 : 255
  }
  return mask
}
