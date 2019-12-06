<template>
  <a-radio-group v-if="tagType=='radio'" @change="handleInput" :value="value" :disabled="disabled">
    <a-radio v-for="(item, key) in dictOptions" :key="key" :value="item.itemName">{{ item.itemCode }}</a-radio>
  </a-radio-group>

  <a-select
    v-else-if="tagType=='select'"
    :placeholder="placeholder"
    :disabled="disabled"
    :value="value"
    @change="handleInput">
    <a-select-option value="">Please Select</a-select-option>
    <a-select-option v-for="(item, key) in dictOptions" :key="key" :value="item.itemCode">
      <span style="display: inline-block;width: 100%" :title=" item.itemName ">
        {{ item.itemName }}
      </span>
    </a-select-option>
  </a-select>
</template>

<script>
import { ajaxGetDictItems, queryTeam } from '@/api/system'

export default {
  name: 'DictSelectTag',
  props: {
    tableName: {
      type: String,
      default: '',
      required: true
    },
    dictCode: {
      type: String,
      default: '',
      required: true
    },
    placeholder: {
      type: String,
      default: '',
      required: false
    },
    triggerChange: {
      type: Boolean,
      default: false,
      required: false
    },
    disabled: {
      type: Boolean,
      default: false,
      required: false
    },
    value: {
      type: String,
      default: '',
      required: false
    },
    type: {
      type: String,
      default: '',
      required: false
    }
  },
  data () {
    return {
      dictOptions: [],
      tagType: ''
    }
  },
  created () {
    // console.log(this.dictCode)
    if (!this.type || this.type === 'list') {
      this.tagType = 'select'
    } else {
      this.tagType = this.type
    }
    // Get dictionary data
    if (this.tableName === 'team') {
      this.initTeamData()
    } else {
      this.initDictData()
    }
  },
  methods: {
    initDictData () {
      // Initialize the dictionary array according to the dictionary Code
      ajaxGetDictItems(this.dictCode, null).then((res) => {
        if (res.success) {
          // console.log(res.result.records)
          this.dictOptions = res.result.records
        }
      })
    },
    initTeamData () {
      // Initialize the dictionary array according to the dictionary Code
      var params = {}
      params.owner = this.$store.getters.userInfo.name
      params.column = 'createTime'
      params.order = 'desc'
      params.pageNo = 0
      params.pageSize = 100
      queryTeam(params).then((res) => {
        if (res.success) {
          console.log(res.result.records)
          if (res.result.records) {
            var dictItems = []
            res.result.records.forEach(function (team) {
              console.log('team=', team)
              var item = {}
              item.itemCode = team.teamName
              item.itemName = team.teamName
              dictItems = [...dictItems, item]
            })
            this.dictOptions = dictItems
          }
        }
      })
    },
    handleInput (e) {
      let val
      if (this.tagType === 'radio') {
        val = e.target.value
      } else {
        val = e
      }
      // console.log(val)
      if (this.triggerChange) {
        this.$emit('change', val)
      } else {
        this.$emit('input', val)
      }
    },
    setCurrentDictOptions (dictOptions) {
      this.dictOptions = dictOptions
    },
    getCurrentDictOptions () {
      return this.dictOptions
    }
  }
}
</script>

<style scoped>
</style>
