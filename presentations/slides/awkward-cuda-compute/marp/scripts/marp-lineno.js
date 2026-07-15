// Marp engine: add line numbers to language-tagged code blocks (plain ``` diagrams are skipped).
let core
try { core = require('@marp-team/marp-core') }
catch { core = require('/usr/local/lib/node_modules/@marp-team/marp-cli/node_modules/@marp-team/marp-core') }
const { Marp } = core

function lineNumbers(md) {
  const orig = md.renderer.rules.fence
  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const html = orig(tokens, idx, options, env, self)
    const lang = tokens[idx].info.trim().split(/\s+/)[0].toLowerCase()
    const skip = new Set(['', 'text', 'plaintext', 'plain', 'output', 'console'])
    if (skip.has(lang)) return html                    // skip diagrams / plain text, number real code only
    return html.replace(/(<code[^>]*>)([\s\S]*?)(<\/code>)/, (m, open, body, close) => {
      const lines = body.replace(/\n$/, '').split('\n')
      return open + lines.map(l => `<span class="cln">${l.length ? l : ' '}</span>`).join('\n') + close
    })
  }
}
module.exports = (opts) => new Marp(opts).use(lineNumbers)
