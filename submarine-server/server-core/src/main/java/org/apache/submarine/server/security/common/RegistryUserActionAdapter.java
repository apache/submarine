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

package org.apache.submarine.server.security.common;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.database.workbench.entity.SysUserEntity;
import org.apache.submarine.server.database.workbench.mappers.SysUserMapper;
import org.pac4j.core.context.JEEContext;
import org.pac4j.core.exception.http.HttpAction;
import org.pac4j.core.http.adapter.JEEHttpActionAdapter;
import org.pac4j.core.profile.CommonProfile;
import org.pac4j.core.profile.ProfileManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Date;
import java.util.Optional;

import static org.apache.submarine.server.rest.workbench.SysUserRestApi.DEFAULT_ADMIN_UID;
import static org.apache.submarine.server.rest.workbench.SysUserRestApi.DEFAULT_CREATE_USER_PASSWORD;

public class RegistryUserActionAdapter<T extends CommonProfile, R extends JEEContext>
        extends JEEHttpActionAdapter {

  private final Logger LOG = LoggerFactory.getLogger(RegistryUserActionAdapter.class);

  @Override
  public T adapt(final HttpAction action, final JEEContext context) {
    super.adapt(action, context);

    // get profile
    ProfileManager<T> manager = new ProfileManager<>(context);
    Optional<T> profile = manager.get(true);

    // every time call back, check if this user is exists
    profile.ifPresent(this::createUndefinedUser);

    return null;
  }

  /**
   * Create a user that does not exist
   */
  protected void createUndefinedUser(CommonProfile ... profiles) {
    LOG.trace("Check user if exists ...");

    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      SysUserMapper sysUserMapper = sqlSession.getMapper(SysUserMapper.class);

      for (CommonProfile profile : profiles) {
        SysUserEntity sysUser = sysUserMapper.getUserByUniqueName(profile.getUsername());
        if (sysUser == null) {
          // if user is undefined, create this user
          LOG.info("Can not find this user, need to create! User profile: {}", profile);
          sysUser = new SysUserEntity();
          sysUser.setUserName(profile.getUsername());
          sysUser.setRealName(profile.getDisplayName());
          sysUser.setPassword(DEFAULT_CREATE_USER_PASSWORD);
          sysUser.setEmail(profile.getEmail());
          sysUser.setPhone(ObjectUtils.identityToString(profile.getAttribute("phone")));
          sysUser.setAvatar(ObjectUtils.identityToString(profile.getPictureUrl()));
          sysUser.setDeleted(0);
          sysUser.setCreateBy(DEFAULT_ADMIN_UID);
          sysUser.setCreateTime(new Date());
          sysUserMapper.add(sysUser);
          sqlSession.commit();
        } else if (sysUser.getDeleted() == 1) {
          LOG.info("Reset this user {} to active", profile.getUsername());
          sysUserMapper.activeUser(sysUser.getId());
          sqlSession.commit();
        }
      }
    } catch (Exception e) {
      LOG.error("Get error when creating user, skip ...", e);
    }
  }
}
