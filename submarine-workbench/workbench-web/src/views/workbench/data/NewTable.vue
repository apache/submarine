
<template>
  <page-view title="Create New Table">

    <a-row slot="extra">
      <p style="margin-left: -88px;font-size: 14px;color: rgba(0,0,0,.65)">{{ description }}</p>
    </a-row>

    <a-card :bordered="false" title="Create A Table In Three Steps">
      <div slot="extra">
        <router-link :to="{ name: 'DataWrapper' }">
          <a-button type="primary" icon="rollback" @click="finish">Cancel</a-button>
        </router-link>
      </div>
      <a-steps class="steps" :current="currentTab">
        <a-step title="Specify Data Source" />
        <a-step title="Specify Table Attributes" />
        <a-step title="Review Table Data" />
      </a-steps>
      <div class="content">
        <step1 v-if="currentTab === 0" @nextStep="nextStep"/>
        <step2 v-if="currentTab === 1" @nextStep="nextStep" @prevStep="prevStep"/>
        <step3 v-if="currentTab === 2" @prevStep="prevStep" @finish="finish"/>
      </div>
    </a-card>
  </page-view>
</template>

<script>
import Step1 from './Step1'
import Step2 from './Step2'
import Step3 from './Step3'
import { mixinDevice } from '@/utils/mixin'
import { PageView } from '@/layouts'
import DetailList from '@/components/tools/DetailList'

const DetailListItem = DetailList.Item

export default {
  name: 'NewTable',
  components: {
    PageView,
    DetailList,
    DetailListItem,
    Step1,
    Step2,
    Step3
  },
  mixins: [mixinDevice],
  data () {
    return {
      currentTab: 0,
      description: 'Create a new table in the database according to the operating instructions.'
    }
  },
  methods: {
    // handler
    nextStep () {
      if (this.currentTab < 2) {
        this.currentTab += 1
      }
    },
    prevStep () {
      if (this.currentTab > 0) {
        this.currentTab -= 1
      }
    },
    finish () {
      this.currentTab = 0
    }
  },
  filters: {
    statusFilter (status) {
      const statusMap = {
        'agree': '成功',
        'reject': '驳回'
      }
      return statusMap[status]
    },
    statusTypeFilter (type) {
      const statusTypeMap = {
        'agree': 'success',
        'reject': 'error'
      }
      return statusTypeMap[type]
    }
  }
}
</script>

<style lang="less" scoped>
  .steps {
    max-width: 750px;
    margin: 16px auto;
  }

  .detail-layout {
    margin-left: 44px;
  }
  .text {
    color: rgba(0, 0, 0, .45);
  }

  .heading {
    color: rgba(0, 0, 0, .85);
    font-size: 20px;
  }

  .no-data {
    color: rgba(0, 0, 0, .25);
    text-align: center;
    line-height: 64px;
    font-size: 16px;

    i {
      font-size: 24px;
      margin-right: 16px;
      position: relative;
      top: 3px;
    }
  }

  .mobile {
    .detail-layout {
      margin-left: unset;
    }
    .text {

    }
    .status-list {
      text-align: left;
    }
  }
</style>
