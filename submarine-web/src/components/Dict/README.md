# DictSelectTag Component introduction

- Get data from the sys_dict table, dictCode format description: Dictionary code
```html
<dict-select-tag v-model="queryParam.sex" placeholder="Please select sex" dictCode="sex"/>
```

v-decorator Function：
```html
<dict-select-tag v-decorator="['sex', {}]" :triggerChange="true" placeholder="Please select sex" dictCode="sex"/>
```
