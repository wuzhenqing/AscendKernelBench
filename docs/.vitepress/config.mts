import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'en-US',
  title: 'AscendKernelBench',
  description: 'Generate, evaluate, and understand LLM-written Ascend C kernels.',
  base: '/AscendKernelBench/',
  lastUpdated: true,
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/AscendKernelBench/favicon.svg' }],
    ['meta', { name: 'theme-color', content: '#087f78' }],
  ],
  sitemap: { hostname: 'https://wuzhenqing.github.io/AscendKernelBench/' },
  themeConfig: {
    logo: '/favicon.svg',
    nav: [
      { text: 'Guide', link: '/guide/getting-started', activeMatch: '^/guide/' },
      { text: 'Reference', link: '/reference/cli', activeMatch: '^/reference/' },
      { text: 'Task authoring', link: '/task_authoring' },
    ],
    sidebar: [
      {
        text: 'Start here',
        items: [
          { text: 'Overview', link: '/' },
          { text: 'Installation & first steps', link: '/guide/getting-started' },
          { text: 'Connect an LLM service', link: '/deploy_llm_service' },
        ],
      },
      {
        text: 'Run the benchmark',
        items: [
          { text: 'Generation & evaluation', link: '/guide/workflows' },
          { text: 'Evaluation protocol', link: '/guide/evaluation' },
          { text: 'Results & metrics', link: '/guide/results' },
          { text: 'Troubleshooting', link: '/guide/troubleshooting' },
        ],
      },
      {
        text: 'Reference',
        items: [
          { text: 'Command line', link: '/reference/cli' },
          { text: 'Configuration & hardware', link: '/reference/configuration' },
          { text: 'Architecture & Python entry points', link: '/reference/architecture' },
        ],
      },
      {
        text: 'Contribute',
        items: [
          { text: 'Write a benchmark task', link: '/task_authoring' },
          { text: 'Maintain & publish the docs', link: '/guide/documentation' },
        ],
      },
    ],
    socialLinks: [{ icon: 'github', link: 'https://github.com/wuzhenqing/AscendKernelBench' }],
    search: { provider: 'local' },
    outline: { level: [2, 3] },
    editLink: {
      pattern: 'https://github.com/wuzhenqing/AscendKernelBench/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'AscendKernelBench contributors',
    },
  },
})
