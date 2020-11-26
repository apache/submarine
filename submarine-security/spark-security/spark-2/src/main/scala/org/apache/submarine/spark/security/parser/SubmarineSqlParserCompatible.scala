package org.apache.submarine.spark.security.parser

import org.apache.spark.sql.catalyst.parser.ParserInterface

class SubmarineSqlParserCompatible(override val delegate: ParserInterface) extends SubmarineSqlParser(delegate: ParserInterface) {

}
