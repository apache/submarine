package org.apache.submarine.server.model.database.service;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.runtime.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.model.database.entities.RegisteredModelEntity;
import org.apache.submarine.server.model.database.mappers.RegisteredModelMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class RegisteredModelService {

  private static final Logger LOG = LoggerFactory.getLogger(RegisteredModelService.class);

  public List<RegisteredModelEntity> selectAll() throws SubmarineRuntimeException {
    LOG.info("Registered model selectAll");
    List<RegisteredModelEntity> registeredModels;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      registeredModels = mapper.selectAll();
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get registered models from database");
    }
    return registeredModels;
  }

  public RegisteredModelEntity select(String name) throws SubmarineRuntimeException {
    LOG.info("Registered Model select:" + name);
    RegisteredModelEntity registeredModel;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      registeredModel = mapper.select(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get registered model name entity from database");
    }
    return registeredModel;
  }

  public RegisteredModelEntity selectWithTag(String name) throws SubmarineRuntimeException {
    LOG.info("Registered Model select with tag:" + name);
    RegisteredModelEntity registeredModel;
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      registeredModel = mapper.selectWithTag(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to get registered model name entity from database");
    }
    return registeredModel;
  }

  public void insert(RegisteredModelEntity registeredModel) throws SubmarineRuntimeException {
    LOG.info("Registered Model insert " + registeredModel.getName());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.insert(registeredModel);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert registered model name entity to database");
    }
  }

  public void update(RegisteredModelEntity registeredModel) throws SubmarineRuntimeException {
    LOG.info("Registered Model update " + registeredModel.getName());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.update(registeredModel);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to update registered model name entity from database");
    }
  }

  public void rename(String name, String newName) throws SubmarineRuntimeException {
    LOG.info("Registered Model rename");
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.rename(name, newName);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to rename registered model name from database");
    }
  }

  public void delete(String name) throws SubmarineRuntimeException {
    LOG.info("Registered Model delete " + name);
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.delete(name);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete registered model entity from database");
    }
  }

  public void deleteAll() throws SubmarineRuntimeException {
    LOG.info("Registered Model delete all");
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelMapper mapper = sqlSession.getMapper(RegisteredModelMapper.class);
      mapper.deleteAll();
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete all registered model entities from database");
    }
  }
}
