component {

  public struct function runAI(required string inputJSON) {
    var tempDir = GetTempDirectory();
    var inputFile = GetTempFile(tempDir, "brain_input_");
    var outputFile = GetTempFile(tempDir, "brain_output_");

    fileWrite(inputFile, arguments.inputJSON);

    try {
      cfexecute(
        name="python",
        arguments="ai_core.py #inputFile# #outputFile#",
        variable="aiResponse",
        timeout="120"
      );

      var output = fileRead(outputFile);
      return deserializeJSON(output);

    } catch (any e) {
      return { "error" = e.message };
    } finally {
      fileDelete(inputFile);
      fileDelete(outputFile);
    }
  }
}
