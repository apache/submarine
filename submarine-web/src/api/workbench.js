import { axios } from '@/utils/request'

const api = {
  workspaceRecent: '/workbench/workspace/recent',
  workspaceRecentFiles: '/workspace/recent/files',
  actuatorList: '/workspace/actuator/list',
  jobList: '/workspace/job/list'
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

export function getActuatorList (parameter) {
  return axios({
    url: api.actuatorList,
    method: 'get',
    params: parameter
  })
}

export function getJobList (parameter) {
  return axios({
    url: api.jobList,
    method: 'get',
    params: parameter
  })
}
