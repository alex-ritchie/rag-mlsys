export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? ''
export const PROFILE: 'local' | 'demo' = (import.meta.env.VITE_PROFILE as 'local' | 'demo' | undefined) ?? 'local'
export const REPO_URL = 'https://github.com/alex-ritchie/rag-mlsys'
export const BOOK = {
  title: 'Machine Learning Systems',
  author: 'Vijay Janapa Reddi',
  url: 'https://mlsysbook.ai/',
  license: 'CC BY-NC-SA 4.0',
  licenseUrl: 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
  source: 'https://github.com/harvard-edge/cs249r_book',
}
