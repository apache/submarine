import { Role } from './role';

export class User {
  avatar = '';
  id = '';
  name = '';
  telephone = '';
  username = '';
  role: Role;
  roleId: number;
  token: string;
}
