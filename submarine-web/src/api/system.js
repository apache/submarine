import { getAction, putAction, postAction } from '@/api/manage'

// 用户管理
const addUser = (params) => postAction('/system/user/add', params)
const editUser = (params) => putAction('/system/user/edit', params)
const queryUserRole = (params) => getAction('/system/user/queryUserRole', params)
const getUserList = (params) => getAction('/system/user/list', params)
const frozenBatch = (params) => putAction('/system/user/frozenBatch', params)
// 验证用户是否存在
const checkOnlyUser = (params) => getAction('/system/user/checkOnlyUser', params)
// 改变密码
const changPassword = (params) => putAction('/system/user/changPassword', params)

// 数据字典
const addDict = (params) => postAction('/sys/dict/add', params)
const editDict = (params) => putAction('/sys/dict/edit', params)
const treeList = (params) => getAction('/sys/dict/treeList', params)
const addDictItem = (params) => postAction('/sys/dictItem/add', params)
const editDictItem = (params) => putAction('/sys/dictItem/edit', params)

const duplicateCheck = (params) => getAction('/sys/duplicateCheck', params)

export {
  addUser,
  editUser,
  queryUserRole,
  getUserList,
  frozenBatch,
  checkOnlyUser,
  changPassword,
  addDict,
  editDict,
  treeList,
  addDictItem,
  editDictItem,
  duplicateCheck
}
