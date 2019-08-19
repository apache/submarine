import { getAction, putAction, postAction } from '@/api/manage'

// 用户管理
const addUser = (params) => postAction('/system/user/add', params)
const editUser = (params) => putAction('/system/user/edit', params)
const queryUserRole = (params) => getAction('/system/user/queryUserRole', params)
const getUserList = (params) => getAction('/system/user/list', params)
// const deleteUser = (params)=>deleteAction("/system/user/delete",params);
// const deleteUserList = (params)=>deleteAction("/system/user/deleteBatch",params);
const frozenBatch = (params) => putAction('/system/user/frozenBatch', params)
// 验证用户是否存在
const checkOnlyUser = (params) => getAction('/system/user/checkOnlyUser', params)
// 改变密码
const changPassword = (params) => putAction('/system/user/changPassword', params)

export {
  addUser,
  editUser,
  queryUserRole,
  getUserList,
  frozenBatch,
  checkOnlyUser,
  changPassword
}
