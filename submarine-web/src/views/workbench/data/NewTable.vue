
<template>
  <page-view title="Create New Table">

    <!--detail-list slot="headerContent" size="small" :col="2" class="detail-layout">
      <detail-list-item term="创建人">曲丽丽</detail-list-item>
      <detail-list-item term="订购产品">XX服务</detail-list-item>
      <detail-list-item term="创建时间">2018-08-07</detail-list-item>
      <detail-list-item term="关联单据"><a>12421</a></detail-list-item>
    </detail-list>
    <a-row slot="extra" class="status-list">
      <a-col :xs="12" :sm="12">
        <div class="text">状态</div>
        <div class="heading">待审批</div>
      </a-col>
      <a-col :xs="12" :sm="12">
        <div class="text">订单金额</div>
        <div class="heading">¥ 568.08</div>
      </a-col>
    </a-row-->
    <!-- actions -->
    <!--template slot="action">
      <a-button-group style="margin-right: 4px;">
        <a-button>操作</a-button>
        <a-button>操作</a-button>
        <a-button><a-icon type="ellipsis"/></a-button>
      </a-button-group>
    </template-->

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
