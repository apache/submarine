package org.apache.submarine.spark.security.parser

import org.apache.spark.sql.catalyst.parser.ParserInterface
import org.apache.spark.sql.types.DataType

class SubmarineSqlParserCompatible(override val delegate: ParserInterface) extends SubmarineSqlParser(delegate: ParserInterface) {

  override def parseMultipartIdentifier(sqlText: String): Seq[String] = {
    delegate.parseMultipartIdentifier(sqlText)
  }

  override def parseRawDataType(sqlText: String): DataType = {
    delegate.parseRawDataType(sqlText)
  }

}
