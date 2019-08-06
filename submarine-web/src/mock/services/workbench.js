import Mock from 'mockjs2'
import { builder, getQueryParameters } from '../util'

const totalCount = 5701

const workspaceRecentFiles = (options) => {
  const parameters = getQueryParameters(options)

  const result = []
  const pageNo = parseInt(parameters.pageNo)
  const pageSize = parseInt(parameters.pageSize)
  const totalPage = Math.ceil(totalCount / pageSize)
  const key = (pageNo - 1) * pageSize
  const next = (pageNo >= totalPage ? (totalCount % pageSize) : pageSize) + 1

  for (let i = 1; i < next; i++) {
    const tmpKey = key + i
    result.push({
      key: tmpKey,
      id: tmpKey,
      commit: 'Commit ' + tmpKey + ' info ...',
      description: 'test' + tmpKey + '.py',
      owner: 'neo',
      status: Mock.mock('@integer(0, 3)'),
      updatedAt: Mock.mock('@datetime'),
      editable: false
    })
  }

  return builder({
    pageSize: pageSize,
    pageNo: pageNo,
    totalCount: totalCount,
    totalPage: totalPage,
    data: result
  })
}

const recentProjects = () => {
  return builder([
    {
      id: 1,
      cover: 'https://gw.alipayobjects.com/zos/rmsportal/zOsKZmFRdUtvpqCImOVY.png',
      title: 'project1',
      description: 'MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字。',
      status: 1,
      updatedAt: '2018-07-26 00:00:00'
    },
    {
      id: 2,
      cover: 'https://gw.alipayobjects.com/zos/rmsportal/zOsKZmFRdUtvpqCImOVY.png',
      title: 'project2',
      description: 'MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字。',
      status: 1,
      updatedAt: '2018-07-26 00:00:00'
    },
    {
      id: 2,
      cover: 'https://gw.alipayobjects.com/zos/rmsportal/zOsKZmFRdUtvpqCImOVY.png',
      title: 'project3',
      description: 'MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字。',
      status: 1,
      updatedAt: '2018-07-26 00:00:00'
    },
    {
      id: 2,
      cover: 'https://gw.alipayobjects.com/zos/rmsportal/zOsKZmFRdUtvpqCImOVY.png',
      title: 'project4',
      description: 'MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字。',
      status: 1,
      updatedAt: '2018-07-26 00:00:00'
    },
    {
      id: 2,
      cover: 'https://gw.alipayobjects.com/zos/rmsportal/zOsKZmFRdUtvpqCImOVY.png',
      title: 'project5',
      description: 'MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字。',
      status: 1,
      updatedAt: '2018-07-26 00:00:00'
    },
    {
      id: 3,
      cover: 'https://gw.alipayobjects.com/zos/rmsportal/dURIMkkrRFpPgTuzkwnB.png',
      title: 'project6',
      description: 'MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字。',
      status: 1,
      updatedAt: '2018-07-26 00:00:00'
    },
    {
      id: 4,
      cover: 'https://gw.alipayobjects.com/zos/rmsportal/sfjbOqnsXXJgNCjCzDBL.png',
      title: 'project7',
      description: 'MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字。',
      status: 1,
      updatedAt: '2018-07-26 00:00:00'
    },
    {
      id: 5,
      cover: 'https://gw.alipayobjects.com/zos/rmsportal/siCrBXXhmvTQGWPNLBow.png',
      title: 'project8',
      description: 'MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字。',
      status: 1,
      updatedAt: '2018-07-26 00:00:00'
    },
    {
      id: 6,
      cover: 'https://gw.alipayobjects.com/zos/rmsportal/ComBAopevLwENQdKWiIn.png',
      title: 'project9',
      description: 'MNIST 数据集训练一个可以识别数据的深度学习模型来帮助识别手写数字。',
      status: 1,
      updatedAt: '2018-07-26 00:00:00'
    }
  ])
}

const news = () => {
  return builder([
    {
      id: 1,
      user: {
        nickname: '@name',
        avatar: 'https://gw.alipayobjects.com/zos/rmsportal/BiazfanxmamNRoxxVxka.png'
      },
      new: {
        title: 'Submarine integration zeppelin',
        url: 'submarine.hadoop.org'
      },
      time: '2018-08-23 14:47:00'
    },
    {
      id: 1,
      user: {
        nickname: '蓝莓酱',
        avatar: 'https://gw.alipayobjects.com/zos/rmsportal/jZUIxmJycoymBprLOUbT.png'
      },
      new: {
        title: 'Submarine supports yarn 2.7.x',
        url: '#'
      },
      time: '2018-08-23 09:35:37'
    },
    {
      id: 1,
      user: {
        nickname: '@name',
        avatar: '@image(64x64)'
      },
      new: {
        title: 'Submarine supports yarn 2.9.x',
        url: '#'
      },
      time: '2017-05-27 00:00:00'
    },
    {
      id: 1,
      user: {
        nickname: '曲丽丽',
        avatar: '@image(64x64)'
      },
      new: {
        title: 'Submarine supports kubernetes',
        url: '#'
      },
      time: '2018-08-23 14:47:00'
    },
    {
      id: 1,
      user: {
        nickname: '@name',
        avatar: '@image(64x64)'
      },
      new: {
        title: 'Submarine 0.1 release',
        url: '#'
      },
      time: '2018-08-23 14:47:00'
    },
    {
      id: 1,
      user: {
        nickname: '@name',
        avatar: '@image(64x64)'
      },
      new: {
        title: 'Submarine 0.2 release',
        url: '#'
      },
      time: '2018-08-23 14:47:00'
    },
    {
      id: 1,
      user: {
        nickname: '@name',
        avatar: '@image(64x64)'
      },
      new: {
        title: 'Submarine 0.3 release',
        url: '#'
      },
      time: '2018-08-23 14:47:00'
    },
    {
      id: 1,
      user: {
        nickname: '曲丽丽',
        avatar: 'https://gw.alipayobjects.com/zos/rmsportal/BiazfanxmamNRoxxVxka.png'
      },
      new: {
        title: 'Submarine 0.4 release',
        url: '#'
      },
      time: '2018-08-23 14:47:00'
    }
  ])
}

const workspaceRecent = () => {
  return builder([{
    'key': 'key-01',
    'title': 'Tensorflow test1',
    'icon': 'folder-open',
    'children': [{
      'key': 'key-01-01',
      'title': 'Python',
      'icon': 'folder-open'
    }, {
      'key': 'key-01-02',
      'title': 'Script',
      'icon': 'folder-open'
    }, {
      'key': 'key-01-03',
      'title': 'Data',
      'icon': 'folder-open',
      'children': [{
        'key': 'key-01-03-01',
        'title': 'Sample',
        'icon': 'folder-open'
      },
      {
        'key': 'key-01-03-02',
        'title': 'Test',
        'icon': 'folder-open'
      }
      ]
    }]
  }, {
    'key': 'key-02',
    'title': 'Tensorflow test2',
    'icon': 'folder-open',
    'children': [{
      'key': 'key-02-01',
      'title': 'Python',
      'icon': 'folder-open'
    }, {
      'key': 'key-02-02',
      'title': 'Script',
      'icon': 'folder-open'
    }, {
      'key': 'key-02-03',
      'title': 'Data',
      'icon': 'folder-open',
      'children': [{
        'key': 'key-02-03-01',
        'title': 'Sample',
        'icon': 'folder-open'
      },
      {
        'key': 'key-02-03-02',
        'title': 'Test',
        'icon': 'folder-open'
      }
      ]
    }]
  }, {
    'key': 'key-03',
    'title': 'Tensorflow test3',
    'icon': 'folder-open',
    'children': [{
      'key': 'key-03-01',
      'title': 'Python',
      'icon': 'folder-open'
    }, {
      'key': 'key-03-02',
      'title': 'Script',
      'icon': 'folder-open'
    }, {
      'key': 'key-03-03',
      'title': 'Data',
      'icon': 'folder-open',
      'children': [{
        'key': 'key-03-03-01',
        'title': 'Sample',
        'icon': 'folder-open'
      },
      {
        'key': 'key-03-03-02',
        'title': 'Test',
        'icon': 'folder-open'
      }
      ]
    }]
  }, {
    'key': 'key-04',
    'title': 'Tensorflow test4',
    'icon': 'folder-open',
    'children': [{
      'key': 'key-04-01',
      'title': 'Python',
      'icon': 'folder-open'
    }, {
      'key': 'key-04-02',
      'title': 'Script',
      'icon': 'folder-open'
    }, {
      'key': 'key-04-03',
      'title': 'Data',
      'icon': 'folder-open',
      'children': [{
        'key': 'key-04-03-01',
        'title': 'Sample',
        'icon': 'folder-open'
      },
      {
        'key': 'key-04-03-02',
        'title': 'Test',
        'icon': 'folder-open'
      }
      ]
    }]
  }])
}

const actuatorList = (options) => {
  const parameters = getQueryParameters(options)

  const result = []
  const pageNo = parseInt(parameters.pageNo)
  const pageSize = parseInt(parameters.pageSize)
  const totalPage = Math.ceil(totalCount / pageSize)
  const key = (pageNo - 1) * pageSize
  const next = (pageNo >= totalPage ? (totalCount % pageSize) : pageSize) + 1

  for (let i = 1; i < next; i++) {
    const tmpKey = key + i
    result.push({
      key: tmpKey,
      id: tmpKey,
      no: 'Spark actuator ' + tmpKey,
      description: 'Spark',
      callNo: Mock.mock('@integer(1, 999)'),
      status: Mock.mock('@integer(0, 3)'),
      updatedAt: Mock.mock('@datetime'),
      progress: {
        // status: 'exception',
        value: Mock.mock('@integer(1, 100)')
      },
      editable: false
    })
  }

  return builder({
    pageSize: pageSize,
    pageNo: pageNo,
    totalCount: totalCount,
    totalPage: totalPage,
    data: result
  })
}

Mock.mock(/\/workbench\/recentProjects/, 'get', recentProjects)
Mock.mock(/\/workbench\/news/, 'get', news)
Mock.mock(/\/workbench\/workspace\/recent/, 'get', workspaceRecent)
Mock.mock(/\/workspace\/recent\/files/, 'get', workspaceRecentFiles)
Mock.mock(/\/workspace\/actuator\/list/, 'get', actuatorList)
