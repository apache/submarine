<template>
  <a-table
    ref="table"
    rowKey="id"
    size="middle"
    bordered
    :columns="columns"
    :dataSource="dataSource"
    :pagination="ipagination"
    :loading="loading">
    <template slot="title">
      <a-button @click="onNewTeam" icon="team">Create New Team</a-button>
    </template>
    <template slot="teamName" slot-scope="text, record">
      <a-input
        v-if="record.editable"
        style="margin: -5px 0"
        :value="text"
        @change="e => onChangeMember(e.target.value, record.id)"
      />
      <template v-else>{{ text }}</template>
    </template>

    <template slot="collaborators" slot-scope="text, record">
      <search-select
        v-if="record.editable"
        :id="record.id"
        :initValue="record.collaborators"
        :disabled="false"
        @change="changeSearchSelect"
        placeholder="Please input user name"/>

      <template v-else>
        <template v-if="record.collaborators.length > 0" v-for="(member) in record.collaborators">
          <a-tooltip placement="top" v-if="member.inviter==0" :key="member.id">
            <template slot="title">
              <span>{{ member.member }} not accepted invitation</span>
            </template>
            <a-tag v-if="member.inviter==0" color="red" :key="member.id">{{ member.member }}</a-tag>
          </a-tooltip>
          <a-tooltip placement="top" v-if="member.inviter==1" :key="member.id">
            <template slot="title">
              <span>{{ member.member }} accept invitation</span>
            </template>
            <a-tag v-if="member.inviter==1" color="blue" :key="member.id">{{ member.member }}</a-tag>
          </a-tooltip>
        </template>
      </template>
    </template>

    <template slot="action" slot-scope="text, record">
      <span>
        <span v-if="record.editable">
          <a @click="() => onSaveMember(record.id)">Save</a> or
          <a-popconfirm title="Sure to cancel?" @confirm="() => onCancelMember(record.id)" okText="Ok" cancelText="Cancel">
            <a>Cancel</a>
          </a-popconfirm>
        </span>

        <span v-else-if="record.owner !== null && record.owner !== '' && record.owner !== login_user.name">
          Permission denied: The owner of the team is <b>{{ record.owner }}</b>
        </span>

        <template v-else>
          <span>
            <a @click="() => onEditMember(record.id)">Edit</a>
          </span>
          <a-divider type="vertical" />
          <a-popconfirm title="Sure to delete?" @confirm="() => onDelMember(record.id)" okText="Ok" cancelText="Cancel">
            <a>Delete</a>
          </a-popconfirm>
        </template>
      </span>
    </template>
  </a-table>
</template>

<script>
import { filterObj } from '@/utils/util'
import { ListMixin } from '@/mixins/ListMixin'
import SearchSelect from '@/components/Dict/SearchSelect.vue'
import { addTeam, editTeam, deleteTeam, duplicateCheck } from '@/api/system'

export default {
  name: 'Team',
  mixins: [ListMixin],
  components: { SearchSelect },
  data () {
    return {
      visible: false,
      // query Param
      queryParam: {
      },
      searchSelectValue: '',
      cacheData: [],
      login_user: {},
      // 表头
      columns: [
        {
          title: 'Team Name',
          align: 'center',
          dataIndex: 'teamName',
          width: 150,
          scopedSlots: { customRender: 'teamName' }
        },
        {
          title: 'Collaborators',
          align: 'left',
          dataIndex: 'collaborators',
          scopedSlots: { customRender: 'collaborators' }
        },
        {
          title: 'Action',
          dataIndex: 'action',
          width: 200,
          align: 'center',
          scopedSlots: { customRender: 'action' }
        }
      ],
      dict: '',
      labelCol: {
        xs: { span: 8 },
        sm: { span: 5 }
      },
      wrapperCol: {
        xs: { span: 16 },
        sm: { span: 19 }
      },
      url: {
        list: '/team/list'
      }
    }
  },
  computed: {
    userInfo () {
      return this.$store.getters.userInfo
    }
  },
  created () {
    this.login_user = this.userInfo
    this.queryParam.owner = this.login_user.name
  },
  methods: {
    getQueryParams () {
      var param = Object.assign({}, this.queryParam, this.isorter)
      param.pageNo = this.ipagination.current
      param.pageSize = this.ipagination.pageSize
      return filterObj(param)
    },
    changeSearchSelect (id, selected, userInfo) {
      // console.log('selected', selected)
      // const changeData = [...this.dataSource]
      const team = this.dataSource.filter(item => item.id === id)[0]
      if (team) {
        // console.log('team', team)
        if (team.collaborators === null) {
          team.collaborators = []
        }
        var deleted = []
        var newAdd = []

        // found deleted team member
        team.collaborators.forEach(function (oldMember) {
          var notFoundNum = 0
          selected.forEach(function (select) {
            if (select.key !== oldMember.id) {
              notFoundNum = notFoundNum + 1
            }
          })
          // var index = selected.indexOf(oldMember.member)
          if (notFoundNum === selected.length) {
            deleted = [...deleted, oldMember.id]
          }
        })

        // found new add team member
        selected.forEach(function (select) {
          var notFoundNum = 0
          team.collaborators.forEach(function (oldMember) {
            if (select.key !== oldMember.id) {
              notFoundNum = notFoundNum + 1
            }
          })
          if (notFoundNum === team.collaborators.length) {
            newAdd = [...newAdd, select]
          }
        })

        // console.log('deleted', deleted)
        // console.log('newAdd', newAdd)
        // delete from dataSource
        var newTeam = Object.assign({}, team)
        newTeam.collaborators = []

        // add not deleted member
        if (team.collaborators) {
          team.collaborators.forEach(function (oldMember) {
            if (deleted.indexOf(oldMember.id) < 0) {
              newTeam.collaborators = [...newTeam.collaborators, oldMember]
            }
          })
        }

        // add new add member
        newAdd.forEach(function (newAddMember) {
          var newMember = {}
          newMember.teamName = team.teamName
          newMember.id = newAddMember.key
          newMember.member = newAddMember.label
          newMember.inviter = 0
          newMember.createBy = userInfo.name
          newMember.updateBy = userInfo.name
          newTeam.collaborators = [...newTeam.collaborators, newMember]
        })
        // console.log('newTeam', newTeam)

        // replace newTeam into dataSource
        this.dataSource.forEach(function (team) {
          if (team.teamName === newTeam.teamName) {
            team.collaborators = newTeam.collaborators
          }
        })
      }
    },
    loadDataSuccess () {
      this.cacheData = this.dataSource.map(item => ({ ...item }))
      // console.log('cacheData', this.cacheData)
    },
    onNewTeam () {
      const checkData = [...this.dataSource]
      // console.log("checkData = ", checkData)
      const target = checkData.filter(item => item.id === '0')[0]
      if (target) {
        // already exist add new team
        return
      }
      const { dataSource } = this
      const newData = {
        id: '0',
        teamName: '',
        owner: this.login_user.name,
        collaborators: [],
        editable: true
      }
      this.dataSource = [...dataSource, newData]
    },
    onChangeMember (value, id) {
      const newData = [...this.dataSource]
      const target = newData.filter(item => id === item.id)[0]
      if (target) {
        // console.log('target = ', target)
        target.teamName = value
        this.dataSource = newData
      }
    },
    onEditMember (id) {
      const newData = [...this.dataSource]
      const target = newData.filter(item => id === item.id)[0]
      if (target) {
        target.editable = true
        this.dataSource = newData
      }
    },
    onSaveMember (id) {
      const newData = [...this.dataSource]
      const target = newData.filter(item => id === item.id)[0]
      // console.log('target = ', target)
      if (target) {
        this.validateTeamName(target, () => {
          if (target.hasOwnProperty('editable')) {
            // need delete editable Property
            delete target['editable']
          }
          // clean collaborators text and value Property
          if (target.collaborators) {
            target.collaborators.forEach((member) => {
              delete member['text']
              delete member['value']
            })
          }

          const that = this
          that.confirmLoading = true
          console.log('target = ', target)
          let obj
          if (target.id === '0') {
            target.createBy = this.login_user.name
            obj = addTeam(target)
          } else {
            target.updateBy = this.login_user.name
            obj = editTeam(target)
          }
          obj.then(res => {
            if (res.success) {
              // console.log('res = ', res)
              that.$message.success(res.message)
              delete target.editable
              if (target.id === '0') {
                target.id = res.result.id
              }
              this.dataSource = newData
              this.cacheData = newData.map(item => ({ ...item }))
              // console.log('dataSource = ', this.dataSource)
              this.loadData(1)
            } else {
              that.$message.warning(res.message)
            }
          }).finally(() => {
            that.confirmLoading = false
          })
        })
      }
    },
    onCancelMember (id) {
      const newData = [...this.dataSource]
      const target = newData.filter(item => id === item.id)[0]
      if (target.id === '0') {
        this.loadData(1)
      } else {
        Object.assign(target, this.cacheData.filter(item => id === item.id)[0])
        delete target.editable
        this.dataSource = newData
      }
    },
    onDelMember (id) {
      const newData = [...this.dataSource]
      const target = newData.filter(item => id === item.id)[0]
      const that = this
      that.confirmLoading = true
      deleteTeam({ id: id }).then(res => {
        if (res.success) {
          that.$message.success(res.message)
          delete target.editable
          const dataSource = [...this.dataSource]
          this.dataSource = dataSource.filter(item => item.id !== id)
        } else {
          that.$message.warning(res.message)
        }
      }).finally(() => {
        that.confirmLoading = false
      })
    },
    validateTeamName (team, callback) {
      const that = this
      if (team.teamName === null || team.teamName === '') {
        that.$message.warning('Team name can not empty!')
        return false
      }
      var params = {
        tableName: 'team',
        fieldName: 'team_name',
        fieldVal: team.teamName
      }
      if (team.id !== '0') {
        params.dataId = team.id
      }
      duplicateCheck(params).then((res) => {
        if (res.success === false) {
          that.$message.warning('Team name already exist!')
        } else {
          callback()
        }
      })
    }
  },
  watch: {
    openKeys (val) {
      console.log('openKeys', val)
    }
  }
}
</script>
