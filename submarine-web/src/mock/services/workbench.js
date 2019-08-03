import Mock from 'mockjs2'
import { builder } from '../util'

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

Mock.mock(/\/workbench\/recentProjects/, 'get', recentProjects)
Mock.mock(/\/workbench\/news/, 'get', news)
