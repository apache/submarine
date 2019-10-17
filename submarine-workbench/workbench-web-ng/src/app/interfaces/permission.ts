export interface PermissionActionEntitySet {
  action: string;
  defaultChecked: boolean;
  describe: string;
}

export type PermissionAction = PermissionActionEntitySet;

export class Permission {
  permissionId = '';
  permissionName = '';
  roleId = '';
  actionList = null;
  actionEntitySet: PermissionActionEntitySet[] = [];
  actions: PermissionAction[] = [];
}
