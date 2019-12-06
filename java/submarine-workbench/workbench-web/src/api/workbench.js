import { axios } from '@/utils/request'

const api = {
  workspaceRecent: '/workbench/workspace/recent',
  workspaceRecentFiles: '/workspace/recent/files',
  actuatorList: '/workspace/actuator/list',
  jobList: '/workspace/job/list',
  dataTables: '/workspace/data/tables',
  databases: '/workspace/data/databases',
  schemaColumnsData: '/workspace/data/columns',
  tableColumns: '/workspace/data/tableColumns',
  sampleData: '/workspace/data/sample'
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

export function getDatabases (parameter) {
  return axios({
    url: api.databases,
    method: 'get',
    params: parameter
  })
}

export function getDataTables (parameter) {
  return axios({
    url: api.dataTables,
    method: 'get',
    params: parameter
  })
}

export function getSchemaColumnsData (parameter) {
  return axios({
    url: api.schemaColumnsData,
    method: 'get',
    params: parameter
  })
}

export function getTableColumns (parameter) {
  return axios({
    url: api.tableColumns,
    method: 'get',
    params: parameter
  })
}

export function getSampleData (parameter) {
  return axios({
    url: api.sampleData,
    method: 'get',
    params: parameter
  })
}
