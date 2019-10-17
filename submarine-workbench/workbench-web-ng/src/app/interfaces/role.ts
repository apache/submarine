import { Permission } from './permission';

export type RoleId = string;

export class Role {
  createTime: number = -1;
  creatorId = '';
  describe = '';
  id = '';
  name = '';
  permissions: Permission[];
}
