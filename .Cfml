<cfset aiResponse = http 
    method="POST" 
    url="http://neomind-backend:5000/infer" 
    result="res">
<cfif res.statusCode EQ 200>
    <cfset resultData = deserializeJSON(res.fileContent)>
    <cfset aiSummary = resultData.summary>
<cfelse>
    <cfset aiSummary = "AI-service unavailable">
</cfif>
