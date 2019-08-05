import { axios } from '@/utils/request'

const api = {
  workspaceRecent: '/workbench/workspace/recent',
  workspaceRecentFiles: '/workspace/recent/files'
}

// export default api

export function getWorkspaceRecent (parameter) {
  return axios({
    url: api.workspaceRecent,
    method: 'get',
    params: parameter
  })
}

export function getWorkspaceRecentFiles (parameter) {
  return axios({
    url: api.workspaceRecentFiles,
    method: 'get',
    params: parameter
  })
}
