<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<template>
  <router-view />
</template>

<script>
export default {
  name: 'WorkspaceLayout',
  data () {
    return {
      description: 'You can click on the project tree directory on the left and the project file list will be displayed on the right.',
      tabs: {
        items: [
          {
            key: '1',
            title: 'Project (2)'
          },
          {
            key: '2',
            title: 'Release (4)'
          },
          {
            key: '3',
            title: 'Training (8)'
          },
          {
            key: '4',
            title: 'Team (3)'
          },
          {
            key: '5',
            title: 'Shared (5)'
          }
        ],
        active: () => {
          switch (this.$route.path) {
            case '/workbench/workspace/project':
              return '1'
            case '/workbench/workspace/release':
              return '2'
            case '/workbench/workspace/training':
              return '3'
            case '/workbench/workspace/team':
              return '4'
            case '/workbench/workspace/shared':
              return '5'
            default:
              return '1'
          }
        },
        callback: (key) => {
          switch (key) {
            case '1':
              this.$router.push('/workbench/workspace/project')
              break
            case '2':
              this.$router.push('/workbench/workspace/release')
              break
            case '3':
              this.$router.push('/workbench/workspace/training')
              break
            case '4':
              this.$router.push('/workbench/workspace/team')
              break
            case '5':
              this.$router.push('/workbench/workspace/shared')
              break
            default:
              this.$router.push('/workbench/workspace/project')
          }
        }
      },
      search: false
    }
  }
}
</script>

<style lang="less" scoped>
  .search-head{
    background-color: #fff;
    margin: -25px -24px -24px;
    .search-input{
      text-align: center;
      margin-bottom: 16px;
    }
  }
  .search-content{
    margin-top: 48px;
  }

  .page-menu-tabs {
    margin-top: 0px;
  }
</style>
