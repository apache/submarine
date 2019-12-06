import { getAction, putAction, postAction, deleteAction } from '@/api/manage'

// 用户管理
const addUser = (params) => postAction('/sys/user/add', params)
const editUser = (params) => putAction('/sys/user/edit', params)
const queryUserRole = (params) => getAction('/sys/user/queryUserRole', params)
const getUserList = (params) => getAction('/sys/user/list', params)
const frozenBatch = (params) => putAction('/sys/user/frozenBatch', params)
// 验证用户是否存在
const checkOnlyUser = (params) => getAction('/sys/user/checkOnlyUser', params)
// change Password
const changePassword = (params) => putAction('/sys/user/changePassword', params)

// 数据字典
const addDict = (params) => postAction('/sys/dict/add', params)
const editDict = (params) => putAction('/sys/dict/edit', params)
const treeList = (params) => getAction('/sys/dict/treeList', params)
const addDictItem = (params) => postAction('/sys/dictItem/add', params)
const editDictItem = (params) => putAction('/sys/dictItem/edit', params)

const duplicateCheck = (params) => getAction('/sys/duplicateCheck', params)
const searchSelect = (tableName, params) => getAction(`/sys/searchSelect/${tableName}`, params)

// department
const addDept = (params) => postAction('/sys/dept/add', params)
const editDept = (params) => putAction('/sys/dept/edit', params)
const queryDepartTreeList = (params) => getAction('/sys/dept/tree', params)
const searchByKeywords = (params) => getAction('/sys/dept/searchBy', params)
const deleteByDepartId = (params) => deleteAction('/sys/dept/delete', params)
const queryIdTree = (params) => getAction('/sys/dept/queryIdTree', params)
const resetParentDept = (params) => putAction('/sys/dept/resetParentDept', params)

// Dict
const ajaxGetDictItems = (code, params) => getAction(`/sys/dictItem/getDictItems/${code}`, params)

// team
const queryTeam = (params) => getAction('/team/list', params)
const addTeam = (params) => postAction('/team/add', params)
const editTeam = (params) => putAction('/team/edit', params)
const deleteTeam = (params) => deleteAction('/team/delete', params)

// project
const queryProject = (params) => getAction('/project/list', params)
const addProject = (params) => postAction('/project/add', params)
const editProject = (params) => putAction('/project/edit', params)
const deleteProject = (params) => deleteAction('/project/delete', params)

export {
  ajaxGetDictItems,
  addUser,
  editUser,
  queryUserRole,
  getUserList,
  frozenBatch,
  checkOnlyUser,
  changePassword,
  addDict,
  editDict,
  treeList,
  addDictItem,
  editDictItem,
  duplicateCheck,
  searchSelect,
  addDept,
  editDept,
  queryDepartTreeList,
  searchByKeywords,
  deleteByDepartId,
  queryIdTree,
  resetParentDept,
  queryTeam,
  addTeam,
  editTeam,
  deleteTeam,
  queryProject,
  addProject,
  editProject,
  deleteProject
}
