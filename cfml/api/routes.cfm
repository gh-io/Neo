<cfsetting enablecfoutputonly="true">
<cfparam name="form.prompt" default="">

<cfset ai = createObject("component", "components.orchestrator")>
<cfset result = ai.runAI(serializeJSON({ prompt = form.prompt }))>

<cfcontent type="application/json">
<cfoutput>#serializeJSON(result)#</cfoutput>
<cfsetting enablecfoutputonly="false">
