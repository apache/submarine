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
        <div className='sidebar-logo'>
          <img className='sidebar-logo-img' src="/assets/logo.png" alt="logo"/>
          <h1>Submarine</h1>
        </div>
        <Menu
          onClick={handleClick}
          theme='dark'

          defaultOpenKeys={['sub1']}
          selectedKeys={[key]}
          mode="inline"
          inlineCollapsed={isCollapsed}
        >
          <Menu.Item key='1' icon={<HomeOutlined />} disabled={true}>
            Home
          </Menu.Item>
          <Menu.Item key='2' icon={<BookOutlined />}>
            Notebook
          </Menu.Item>
          <Menu.Item key='3' icon={<ExperimentOutlined />}>
            Experiment
          </Menu.Item>
          <Menu.Item key='4' icon={<FileOutlined />}>
            Template
          </Menu.Item>
          <Menu.Item key='5' icon={<CodepenOutlined />}>
            Environment
          </Menu.Item>
          <SubMenu key="sub2" icon={<SettingOutlined />} title="Manager">
            <Menu.Item key="6">User</Menu.Item>
            <Menu.Item key="7">Data dict</Menu.Item>
            <Menu.Item key="8">Department</Menu.Item>
          </SubMenu>
          <Menu.Item key='9' icon={<InboxOutlined />}>
            Model
          </Menu.Item>
          <Menu.Item key='10' icon={<BarChartOutlined />} disabled={true}>
            Data
          </Menu.Item>
          <Menu.Item key='11' icon={<DesktopOutlined />} disabled={true}>
            Workspace
          </Menu.Item>
          <Menu.Item key='12' icon={<ApiOutlined />} disabled={true}>
            Interpreter
          </Menu.Item>
        </Menu>
      </Sider>
      
    </Layout>
  )
}

export default Sidebar;