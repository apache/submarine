/*!
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import { useState } from 'react'
import { Layout, Menu } from 'antd';
import SubMenu from 'antd/lib/menu/SubMenu';
import {
  HomeOutlined,
  ExperimentOutlined,
  BookOutlined,
  FileOutlined,
  CodepenOutlined,
  InboxOutlined,
  SettingOutlined,
  BarChartOutlined,
  DesktopOutlined,
  ApiOutlined
} from '@ant-design/icons';
import Sider from 'antd/lib/layout/Sider';
import './Sidebar.scss'
import { Link } from 'react-router-dom'

function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [key, setKey] = useState("")

  const handleClick = (e: { key: string; }) => setKey(key => e.key);

  return (
    <Layout>
      <Sider
        className='menu-sidebar'
        width='256px'
        breakpoint='md'
        onCollapse={(collapsed, type) => {
          setIsCollapsed(() => collapsed);
        }}>
        <Link to='/experiment' onClick={() => setKey("experiment")}>
          <div className='sidebar-logo'>
            <img className='sidebar-logo-img' src="/assets/logo.png" alt="logo" />
            <h1>Submarine</h1>
          </div>
        </Link>
        <Menu
          onClick={handleClick}
          theme='dark'

          defaultOpenKeys={['sub1']}
          selectedKeys={[key]}
          mode="inline"
          inlineCollapsed={isCollapsed}
        >
          <Menu.Item key='home' icon={<HomeOutlined />} disabled={true}>
            <Link to='/home'>
              Home
            </Link>
          </Menu.Item>
          <Menu.Item key='notebook' icon={<BookOutlined />}>
            <Link to='/notebook'>
              Notebook
            </Link>
          </Menu.Item>
          <Menu.Item key='experiment' icon={<ExperimentOutlined />}>
            <Link to='/experiment'>
              Experiment
            </Link>
          </Menu.Item>
          <Menu.Item key='template' icon={<FileOutlined />}>
            <Link to='/template'>
              Template
            </Link>
          </Menu.Item>
          <Menu.Item key='environment' icon={<CodepenOutlined />}>
            <Link to='/environment'>
              Environment
            </Link>
          </Menu.Item>
          <SubMenu key="sub2" icon={<SettingOutlined />} title="Manager">
            <Menu.Item key="user">
              <Link to='/user'>
                User
              </Link>
            </Menu.Item>
            <Menu.Item key="dataDict">
              <Link to='/dataDict'>
                Data Dict
              </Link>
            </Menu.Item>
            <Menu.Item key="department">
              <Link to='/department'>
                Department
              </Link>
            </Menu.Item>
          </SubMenu>
          <Menu.Item key='model' icon={<InboxOutlined />}>
            <Link to='/model'>
              Model
            </Link>
          </Menu.Item>
          <Menu.Item key='data' icon={<BarChartOutlined />} disabled={true}>
            <Link to='/data'>
              Data
            </Link>
          </Menu.Item>
          <Menu.Item key='workspace' icon={<DesktopOutlined />} disabled={true}>
            <Link to='/workspace'>
              Workspace
            </Link>
          </Menu.Item>
          <Menu.Item key='interpreter' icon={<ApiOutlined />} disabled={true}>
            <Link to='/interpreter'>
              Interpreter
            </Link>
          </Menu.Item>
        </Menu>
      </Sider>

    </Layout>
  )
}

export default Sidebar;