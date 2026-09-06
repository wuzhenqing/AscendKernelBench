// Check the actual output, including fragments VitePress does not validate.
import { readFile, readdir, stat } from 'node:fs/promises'
import path from 'node:path'

const root = path.resolve('docs/.vitepress/dist')
const origin = 'https://docs.invalid'
const base = '/AscendKernelBench/'

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const nested = await Promise.all(entries.map((entry) => {
    const filename = path.join(directory, entry.name)
    return entry.isDirectory() ? walk(filename) : [filename]
  }))
  return nested.flat()
}

function decode(value) {
  return value.replace(/&(?:amp|quot|apos|lt|gt|#\d+|#x[\da-f]+);/gi, (entity) => {
    const named = { '&amp;': '&', '&quot;': '"', '&apos;': "'", '&lt;': '<', '&gt;': '>' }
    if (entity in named) return named[entity]
    const hex = entity.toLowerCase().startsWith('&#x')
    return String.fromCodePoint(parseInt(entity.slice(hex ? 3 : 2, -1), hex ? 16 : 10))
  })
}

const files = await walk(root)
const pages = new Map()
for (const file of files.filter((file) => file.endsWith('.html'))) {
  const html = (await readFile(file, 'utf8')).replace(/<!--[\s\S]*?-->/g, '')
  const tags = html.match(/<[a-z][^>]*>/gi) ?? []
  const ids = new Set(tags.flatMap((tag) =>
    [...tag.matchAll(/\bid=["']([^"']*)["']/g)].map((match) => decode(match[1]))))
  pages.set(file, { tags, ids })
}

const errors = []
let checked = 0
for (const [file, { tags }] of pages) {
  const relative = path.relative(root, file).split(path.sep).join('/')
  const pageUrl = new URL(base + relative, origin)
  for (const tag of tags) {
    for (const [, raw] of tag.matchAll(/\b(?:href|src)=["']([^"']*)["']/g)) {
      const url = new URL(decode(raw), pageUrl)
      if (url.origin !== origin) continue
      checked++
      if (!url.pathname.startsWith(base)) {
        errors.push(`${relative}: URL escapes the deployment base: ${raw}`)
        continue
      }
      let target = path.resolve(root, decodeURIComponent(url.pathname.slice(base.length)))
      if (target !== root && !target.startsWith(root + path.sep)) {
        errors.push(`${relative}: URL escapes the output directory: ${raw}`)
        continue
      }
      if (url.pathname.endsWith('/')) target = path.join(target, 'index.html')
      const info = await stat(target).catch(() => null)
      if (!info?.isFile()) {
        errors.push(`${relative}: Missing local target: ${raw}`)
      } else if (url.hash && pages.has(target)) {
        const fragment = decodeURIComponent(url.hash.slice(1))
        if (fragment && !pages.get(target).ids.has(fragment)) {
          errors.push(`${relative}: Missing fragment: ${raw}`)
        }
      }
    }
  }
}

if (errors.length) {
  console.error([...new Set(errors)].join('\n'))
  process.exitCode = 1
} else {
  console.log(`Checked ${pages.size} HTML pages and ${checked} local links/assets; no broken targets or fragments.`)
}
