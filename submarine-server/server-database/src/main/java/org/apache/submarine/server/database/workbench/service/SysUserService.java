/*
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
package org.apache.submarine.server.database.workbench.service;

import org.apache.ibatis.session.RowBounds;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.database.workbench.entity.SysUserEntity;
import org.apache.submarine.server.database.workbench.mappers.SysUserMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

public class SysUserService {
  private static final Logger LOG = LoggerFactory.getLogger(SysUserService.class);

  private static final String GET_USER_BY_NAME_STATEMENT
      = "org.apache.submarine.server.database.workbench.mappers.SysUserMapper.getUserByName";

  // default user is admin
  public static final String DEFAULT_ADMIN_UID = "e9ca23d68d884d4ebb19d07889727dae";
  // default password is `password` by angular markAsDirty method
  public static final String DEFAULT_CREATE_USER_PASSWORD = "5f4dcc3b5aa765d61d8327deb882cf99";

  public SysUserEntity getUserByName(String name, String password) throws Exception {
    SysUserEntity sysUser = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      HashMap<String, Object> mapParams = new HashMap<>();
      mapParams.put("name", name);
      mapParams.put("password", password);

      Map<String, Object> params = new HashMap<>();
      params.put("mapParams", mapParams);

      sysUser = sqlSession.selectOne(GET_USER_BY_NAME_STATEMENT, params);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return sysUser;
  }

  /**
   * Get user by unique name
   */
  public SysUserEntity getUserByName(String name) {
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper sysUserMapper = sqlSession.getMapper(SysUserMapper.class);
      return sysUserMapper.getUserByUniqueName(name);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw e;
    }
  }

  /**
   * Get or create undefined user:
   * 1. If present, determine if reactivation is required
   * 1. If not present, create user
   */
  public SysUserEntity getOrCreateUser(String username, Supplier<SysUserEntity> entitySupplier) {
    LOG.trace("Check user if exists ...");
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper sysUserMapper = sqlSession.getMapper(SysUserMapper.class);
      SysUserEntity sysUser = sysUserMapper.getUserByUniqueName(username);
      if (sysUser == null) {
        // if user is undefined, create this user
        sysUser = entitySupplier.get();
        LOG.info("Can not find this user, need to create! User entity: {}", sysUser);
        sysUserMapper.add(sysUser);
        sqlSession.commit();
      } else if (sysUser.getDeleted() == 1) {
        LOG.info("Reset this user {} to active", username);
        sysUserMapper.activeUser(sysUser.getId());
        sqlSession.commit();
      }
      return sysUser;
    } catch (Exception e) {
      LOG.error("Get error when creating user, skip ...", e);
      return null;
    }
  }

  public SysUserEntity login(HashMap<String, String> mapParams) throws Exception {
    SysUserEntity sysUser = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper sysUserMapper = sqlSession.getMapper(SysUserMapper.class);
      sysUser = sysUserMapper.login(mapParams);
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return sysUser;
  }

  public List<SysUserEntity> queryPageList(String userName,
                                           String email,
                                           String deptCode,
                                           String column,
                                           String field,
                                           int pageNo,
                                           int pageSize) throws Exception {
    LOG.info("SysUserService::queryPageList userName:{}, email:{}, deptCode:{}, " +
            "column:{}, field:{}, pageNo:{}, pageSize:{}",
        userName, email, deptCode, column, field, pageNo, pageSize);

    List<SysUserEntity> list = null;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper sysUserMapper = sqlSession.getMapper(SysUserMapper.class);
      Map<String, Object> where = new HashMap<>();
      where.put("userName", userName);
      where.put("email", email);
      where.put("deptCode", deptCode);
      list = sysUserMapper.selectAll(where, new RowBounds(pageNo, pageSize));
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return list;
  }

  public boolean add(SysUserEntity sysUser) throws Exception {
    LOG.info("add({})", sysUser.toString());

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper userMapper = sqlSession.getMapper(SysUserMapper.class);
      userMapper.add(sysUser);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean edit(SysUserEntity sysUser) throws Exception {
    LOG.info("edit({})", sysUser.toString());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper userMapper = sqlSession.getMapper(SysUserMapper.class);
      userMapper.updateBy(sysUser);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean delete(String id) throws Exception {
    LOG.info("delete({})", id.toString());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper userMapper = sqlSession.getMapper(SysUserMapper.class);
      userMapper.deleteById(id);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }

  public boolean changePassword(SysUserEntity user) throws Exception {
    LOG.info("changePassword({})", user.toString());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper userMapper = sqlSession.getMapper(SysUserMapper.class);
      userMapper.changePassword(user);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new Exception(e);
    }
    return true;
  }
}
