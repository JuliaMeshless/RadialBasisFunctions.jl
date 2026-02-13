import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import { mathjaxPlugin } from './mathjax-plugin'
import footnote from "markdown-it-footnote";
import path from 'path'

const mathjax = mathjaxPlugin()

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: '/',// defined defined defined defined defined defined by define define defined
}

const navTemp = {
  nav: [
    { text: 'Home', link: '/index' },
    { text: 'Getting Started', link: '/getting_started' },
    { text: 'Autodiff', link: '/autodiff' },
    { text: 'Quick Reference', link: '/quickref' },
    { text: 'Theory', link: '/theory' },
    { text: 'Internals', link: '/internals' },
    { text: 'API', link: '/api' }
  ],
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
    base: '/',// defined defined defined defined defined defined by define define defined
    title: 'RadialBasisFunctions.jl',
    description: 'Documentation for RadialBasisFunctions.jl',
    lastUpdated: true,
    cleanUrls: true,
    outDir: '../1',
    head: [
      ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
      ['script', {src: `${baseTemp.base}siteinfo.js`}]
    ],

    vite: {
      plugins: [
        mathjax.vitePlugin,
      ],
      define: {
        __DEPLOY_ABSPATH__: JSON.stringify('/'),
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '../components')
        }
      },
      optimizeDeps: {
        exclude: [
          '@nolebase/vitepress-plugin-enhanced-readabilities/client',
          'vitepress',
          '@nolebase/ui',
        ],
      },
      ssr: {
        noExternal: [
          '@nolebase/vitepress-plugin-enhanced-readabilities',
          '@nolebase/ui',
        ],
      },
    },
    markdown: {
      config(md) {
        md.use(tabsMarkdownPlugin);
        md.use(footnote);
        mathjax.markdownConfig(md);
      },
      theme: {
        light: "github-light",
        dark: "github-dark"
      }
    },
    themeConfig: {
      outline: 'deep',
      logo: { src: '/logo.svg', width: 24, height: 24},
      search: {
        provider: 'local',
        options: {
          detailedView: true
        }
      },
      nav,
      sidebar: [
        { text: 'Home', link: '/index' },
        { text: 'Getting Started', link: '/getting_started' },
        { text: 'Autodiff', link: '/autodiff' },
        { text: 'Quick Reference', link: '/quickref' },
        { text: 'Theory', link: '/theory' },
        { text: 'Internals', link: '/internals' },
        { text: 'API', link: '/api' }
      ],
      editLink: { pattern: "https://github.com/JuliaMeshless/RadialBasisFunctions.jl/edit/main/docs/src/:path" },
      socialLinks: [
        { icon: 'github', link: 'https://github.com/JuliaMeshless/RadialBasisFunctions.jl' }
      ],
      footer: {
        message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
        copyright: `© Copyright ${new Date().getUTCFullYear()}.`
      }
    }
  })
