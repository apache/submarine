<template>
  <a-select
    mode="multiple"
    style="width: 100%"
    labelInValue
    :placeholder="placeholder"
    :disabled="disabled"
    :id="id"
    :filterOption="false"
    :defaultValue="initSelectValue"
    @search="fetchUser"
    @change="handleChange"
    :notFoundContent="fetching ? undefined : null"
  >
    <a-spin v-if="fetching" slot="notFoundContent" size="small"/>
    <a-select-option v-for="d in data" :key="d.value">{{ d.text }}</a-select-option>
  </a-select>
</template>
<script>

import debounce from 'lodash/debounce'
import { searchSelect } from '@/api/system'

export default {
  name: 'SearchSelect',
  props: {
    placeholder: {
      type: String,
      default: 'Please input keyword',
      required: false
    },
    initValue: {
      type: String,
      default: '',
      required: false
    },
    id: {
      type: String,
      default: '',
      required: false
    },
    disabled: {
      type: Boolean,
      default: false,
      required: false
    }
  },
  watch: {
    initValue: {
      immediate: true,
      handler (val) {
        console.log('val = ', val)
        if (val && this.initSelectValue.length === 0) {
          for (var i = 0; i < val.length; i++) {
            var option = {}
            option.key = val[i].id
            option.value = val[i].id
            option.label = val[i].member
            option.text = val[i].member
            // console.log('option = ', option)
            this.initSelectValue.push(option)
          }
          // console.log('this.initSelectValue = ', this.initSelectValue)
          this.data = Object.assign({}, this.initSelectValue)
        }
      }
    }
  },
  data () {
    this.lastFetchId = 0
    this.fetchUser = debounce(this.fetchUser, 800)
    return {
      data: [],
      value: [],
      initSelectValue: [],
      fetching: false
    }
  },
  methods: {
    fetchUser (value) {
      // console.log('fetching user', value)
      if (value === null || value === '') {
        return
      }

      this.lastFetchId += 1
      const fetchId = this.lastFetchId
      this.data = []
      this.fetching = true

      searchSelect('sys_user', { keyword: value }).then((res) => {
        if (res.success) {
          if (fetchId !== this.lastFetchId) { // for fetch callback order
            return
          }

          const data = res.result.records.map(user => ({
            text: `${user.userName}`,
            value: user.id
          }))
          this.data = data
          this.fetching = false
        }
      })
    },
    handleChange (value) {
      this.$emit('change', this.id, value)
    }
  }
}
</script>
