def parse_time(exec_time):
    """ Parse execution time into h:m:s format """
    hours = int( exec_time / 3600 ) 
    exec_time -= hours*3600
    minutes = int( exec_time / 60 )
    exec_time -= minutes*60
    seconds = int( exec_time )
    exec_time -= seconds
    millisecs = int( exec_time * 1000 )
    return (str(hours)+'h '+str(minutes)+'m '
            +str(seconds)+'s '+str(millisecs)+'ms')

def output_solution(solution, ofilename, decorate=False):
    # Content to write to the output file
    first = True
    content = ""
    for node in solution:
        if decorate:
            if first:
                first = False
                content += ""
            else:
                content += " -> "
            content += "Node " + str(node)
        else:
            if first:
                first = False
                content += ""
            else:
                content += ','
            content += str(node)
            
    if decorate:
        content += " -> Node " + str(solution[0])+' EOF\n'
    else:
        content += ','+str(solution[0])+' EOF\n'
        
    with open(ofilename, 'w') as ofile:
        ofile.write(content)