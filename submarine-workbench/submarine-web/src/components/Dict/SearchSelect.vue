<template>
  <a-select
    mode="multiple"
    labelInValue
    :value="value"
    :placeholder="placeholder"
    :disabled="disabled"
    style="width: 100%"
    :filterOption="false"
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
    disabled: {
      type: Boolean,
      default: false,
      required: false
    }
  },
  data () {
    this.lastFetchId = 0
    this.fetchUser = debounce(this.fetchUser, 800)
    return {
      data: [],
      value: [],
      fetching: false
    }
  },
  methods: {
    fetchUser (value) {
      console.log('fetching user', value)
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
      Object.assign(this, {
        value,
        data: [],
        fetching: false
      })
      this.$emit('change', this.value)
    }
  }
}
</script>
